/*
 * Copyright (c) 2006, 2007 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2007, 2008 Mellanox Technologies. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <linux/mm.h>
#include <linux/dma-mapping.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/errno.h>
#include <linux/export.h>
#include <linux/hugetlb.h>
#include <linux/atomic.h>
#include <linux/pci.h>
#include <linux/kernel.h>

#include "nv-p2p.h"
#include <rdma/peer_mem.h>

#define DRV_NAME	"nv_mem"
#define DRV_VERSION	"1.3-0"
#define DRV_RELDATE	__DATE__

MODULE_AUTHOR("Yishai Hadas");
MODULE_DESCRIPTION("NVIDIA GPU memory plug-in");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_VERSION(DRV_VERSION);

#define peer_err(FMT, ARGS...) printk(KERN_ERR   DRV_NAME " %s:%d " FMT, __FUNCTION__, __LINE__, ## ARGS)

static int enable_dbg = 0;

#define peer_dbg(FMT, ARGS...)                                          \
        do {                                                            \
                if (enable_dbg /*&& printk_ratelimit()*/)		\
                        printk(KERN_ERR DRV_NAME " DBG %s:%d " FMT, __FUNCTION__, __LINE__, ## ARGS); \
        } while(0)

module_param(enable_dbg, int, 0000);     // 定义模块参数，用于启用调试追踪
MODULE_PARM_DESC(enable_dbg, "enable debug tracing");  // 模块参数描述

#ifndef NVIDIA_P2P_MAJOR_VERSION_MASK
#define NVIDIA_P2P_MAJOR_VERSION_MASK   0xffff0000
#endif

#ifndef NVIDIA_P2P_MINOR_VERSION_MASK
#define NVIDIA_P2P_MINOR_VERSION_MASK   0x0000ffff
#endif

#ifndef NVIDIA_P2P_MAJOR_VERSION
#define NVIDIA_P2P_MAJOR_VERSION(v)	\
	(((v) & NVIDIA_P2P_MAJOR_VERSION_MASK) >> 16)
#endif

#ifndef NVIDIA_P2P_MINOR_VERSION
#define NVIDIA_P2P_MINOR_VERSION(v)	\
	(((v) & NVIDIA_P2P_MINOR_VERSION_MASK))
#endif

/*
 *	Note: before major version 2, struct dma_mapping had no version field,
 *	so it is not possible to check version compatibility. In this case
 *	let us just avoid dma mappings altogether.
 */
#if defined(NVIDIA_P2P_DMA_MAPPING_VERSION) &&	\
	(NVIDIA_P2P_MAJOR_VERSION(NVIDIA_P2P_DMA_MAPPING_VERSION) >= 2)
#pragma message("Enable nvidia_p2p_dma_map_pages support")
#define NV_DMA_MAPPING 1
#else
#define NV_DMA_MAPPING 0
#endif

#ifndef READ_ONCE
#define READ_ONCE(x) ACCESS_ONCE(x)
#endif

#ifndef WRITE_ONCE
#define WRITE_ONCE(x, val) ({ ACCESS_ONCE(x) = (val); })
#endif

#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    ((u64)1 << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
#define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)


invalidate_peer_memory mem_invalidate_callback;  // 定义无效化对等内存的回调函数
static void *reg_handle = NULL;  // 注册句柄
static void *reg_handle_nc = NULL;  // 非缓存注册句柄

// NVIDIA GPU内存上下文结构体
struct nv_mem_context {
	// NVIDIA对等内存页表
	struct nvidia_p2p_page_table *page_table;
#if NV_DMA_MAPPING
	// NVIDIA对等DMA映射
	struct nvidia_p2p_dma_mapping *dma_mapping;
#endif
#ifndef PEER_MEM_U64_CORE_CONTEXT
	// 核心上下文
	void *core_context;
#else
	
	u64 core_context;
#endif
	u64 page_virt_start;	// 虚拟页起始地址
	u64 page_virt_end;	// 虚拟页结束地址
	size_t mapped_size;	// 映射大小
	unsigned long npages;	// 页数
	unsigned long page_size;	 // 页大小
	struct task_struct *callback_task; 	// 回调任务
	int sg_allocated;	// 分散/聚集列表分配标志
	struct sg_table sg_head;	// 分散/聚集列表头部
};

// 判断是否支持持久页面
static inline int nv_support_persistent_pages(void)
{
#ifdef NVIDIA_P2P_CAP_PERSISTENT_PAGES
	return !!(nvidia_p2p_cap_persistent_pages);
#else
	return 0;
#endif
}

// 释放对等内存回调函数
static void nv_get_p2p_free_callback(void *data)
/*
 * 总体功能：对等内存释放回调函数，用于释放NVIDIA GPU内存插件中使用的对等内存资源。
 * 通过该函数，可以安全地释放之前申请的对等内存资源，包括页表和DMA映射。
 */
{
	int ret = 0;
	struct nv_mem_context *nv_mem_context = (struct nv_mem_context *)data;
	struct nvidia_p2p_page_table *page_table = NULL; // 初始化页表指针为空
#if NV_DMA_MAPPING
	struct nvidia_p2p_dma_mapping *dma_mapping = NULL; // 初始化DMA映射指针为空
#endif

	__module_get(THIS_MODULE); // 增加模块引用计数，防止模块被卸载
	if (!nv_mem_context) { // 如果nv_mem_context为空指针
		peer_err("nv_get_p2p_free_callback -- invalid nv_mem_context\n"); // 打印错误信息
		goto out; // 跳转到结束标签，执行清理工作
	}

	if (!nv_mem_context->page_table) { // 如果页表指针为空
		peer_err("nv_get_p2p_free_callback -- invalid page_table\n"); // 打印错误信息
		goto out; // 跳转到结束标签，执行清理工作
	}

	/* Save page_table locally to prevent it being freed as part of nv_mem_release
	    in case it's called internally by that callback.
	*/
	// 保存页表以防止在回调期间释放
	page_table = nv_mem_context->page_table;

#if NV_DMA_MAPPING
	if (!nv_mem_context->dma_mapping) { // 如果DMA映射指针为空
		peer_err("nv_get_p2p_free_callback -- invalid dma_mapping\n");
		goto out;
	}
	// 设置DMA映射指针
	dma_mapping = nv_mem_context->dma_mapping;
#endif

	/* For now don't set nv_mem_context->page_table to NULL, 
	  * confirmed by NVIDIA that inflight put_pages with valid pointer will fail gracefully.
	*/
	// 打印调试信息
        peer_dbg("calling mem_invalidate_callback\n");
	// 设置回调任务为当前任务
	nv_mem_context->callback_task = current;
	// 调用无效化对等内存的回调函数
	(*mem_invalidate_callback) (reg_handle, nv_mem_context->core_context);
	// 清空回调任务
	nv_mem_context->callback_task = NULL;

#if NV_DMA_MAPPING
	ret = nvidia_p2p_free_dma_mapping(dma_mapping);  // 释放DMA映射
	if (ret)
                peer_err("nv_get_p2p_free_callback -- error %d while calling nvidia_p2p_free_dma_mapping()\n", ret);
#endif
	ret = nvidia_p2p_free_page_table(page_table);  // 释放页表
	if (ret)
		peer_err("nv_get_p2p_free_callback -- error %d while calling nvidia_p2p_free_page_table()\n", ret);

out:
	module_put(THIS_MODULE);  // 释放模块引用计数，允许模块被卸载
	return;

}

/* At that function we don't call IB core - no ticket exists */
static void nv_mem_dummy_callback(void *data)
/*
 * 主要功能：虚拟内存回调函数，用于释放对等内存上下文中的页表资源。
 * 该函数被设计为一个虚拟的回调函数，用于在特定情况下释放对等内存资源，
 * 但实际上并未执行真正的回调操作，而是直接释放了传入的对等内存上下文中的页表资源。
 */
{
	// 将传入的数据转换为对等内存上下文结构体指针
	struct nv_mem_context *nv_mem_context = (struct nv_mem_context *)data;
	// 初始化返回值
	int ret = 0;
	// 增加模块引用计数，防止模块被卸载
	__module_get(THIS_MODULE);
	// 释放传入对等内存上下文中的页表资源
	ret = nvidia_p2p_free_page_table(nv_mem_context->page_table);
	
	if (ret)
		peer_err("nv_mem_dummy_callback -- error %d while calling nvidia_p2p_free_page_table()\n", ret);

	module_put(THIS_MODULE);
	return;
}

/* acquire return code: 1 mine, 0 - not mine */
static int nv_mem_acquire(unsigned long addr, size_t size, void *peer_mem_private_data,
					char *peer_mem_name, void **client_context)
/*
 * 主要功能：获取对等内存资源的函数，用于申请并获取GPU之间共享的对等内存区域。
 * 当调用该函数时，会在给定的内存地址范围内申请对等内存资源，并将相关信息保存在客户端上下文中。
 * 如果成功获取到对等内存资源，返回值为1，同时将客户端上下文指针指向分配的内存区域；
 * 如果获取失败，则返回值为0。
 */

{

	int ret = 0;
	// 定义对等内存上下文结构体指针
	struct nv_mem_context *nv_mem_context;
	// 通过内核内存分配函数kzalloc分配对等内存上下文结构体内存空间
	nv_mem_context = kzalloc(sizeof *nv_mem_context, GFP_KERNEL);
	// 如果内存分配失败， 返回0，表示获取对等内存资源失败
	if (!nv_mem_context)
		/* Error case handled as not mine */
		return 0;
	// 计算页表起始和结束地址
	nv_mem_context->page_virt_start = addr & GPU_PAGE_MASK;
	nv_mem_context->page_virt_end   = (addr + size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
	nv_mem_context->mapped_size  = nv_mem_context->page_virt_end - nv_mem_context->page_virt_start;
	// 调用NVIDIA函数获取对等内存页表
	ret = nvidia_p2p_get_pages(0, 0, nv_mem_context->page_virt_start, nv_mem_context->mapped_size,
			&nv_mem_context->page_table, nv_mem_dummy_callback, nv_mem_context);

	if (ret < 0)
		goto err;
	// 调用NVIDIA函数释放对等内存页表（模拟操作）
	ret = nvidia_p2p_put_pages(0, 0, nv_mem_context->page_virt_start,
				   nv_mem_context->page_table);
	if (ret < 0) {
		/* Not expected, however in case callback was called on that buffer just before
		    put pages we'll expect to fail gracefully (confirmed by NVIDIA) and return an error.
		*/
		peer_err("nv_mem_acquire -- error %d while calling nvidia_p2p_put_pages()\n", ret);
		goto err;
	}

	/* 1 means mine */
	// 将客户端上下文指针指向分配的对等内存上下文结构体
	*client_context = nv_mem_context;
	__module_get(THIS_MODULE);
	return 1;

err:
	kfree(nv_mem_context);

	/* Error case handled as not mine */
	return 0;
}

static int nv_dma_map(struct sg_table *sg_head, void *context,
			      struct device *dma_device, int dmasync,
			      int *nmap)
/*
 * 函数名：nv_dma_map()
 * 功能：执行DMA映射操作，将GPU虚拟内存页映射到DMA设备的散射/聚集列表（scatter/gather list，即sg表）中
 * 参数：
 *   - sg_head: 指向DMA设备的sg表的指针
 *   - context: 指向nv_mem_context结构体的指针，包含了DMA映射操作所需的上下文信息
 *   - dma_device: 指向DMA设备的指针
 *   - dmasync: DMA同步标志，指示是否需要进行DMA同步操作
 *   - nmap: 用于返回成功映射的页数的指针
 * 返回值：成功返回0，否则返回负值错误码
 */
{
	int i, ret;
	struct scatterlist *sg;
	struct nv_mem_context *nv_mem_context =
		(struct nv_mem_context *) context;
	struct nvidia_p2p_page_table *page_table = nv_mem_context->page_table;

	// 检查页表的页面大小是否为64KB
	if (page_table->page_size != NVIDIA_P2P_PAGE_SIZE_64KB) {
		peer_err("nv_dma_map -- assumption of 64KB pages failed size_id=%u\n",
					nv_mem_context->page_table->page_size);
		return -EINVAL;
	}

#if NV_DMA_MAPPING
	{
		# 这个分支主要用于处理 DMA 映射的情况，包括执行 DMA 映射操作、检查 DMA 设备的有效性、分配 SG 表等
		struct nvidia_p2p_dma_mapping *dma_mapping;
		struct pci_dev *pdev = to_pci_dev(dma_device);
		// 检查DMA设备是否有效
		if (!pdev) {
			peer_err("nv_dma_map -- invalid pci_dev\n");
			return -EINVAL;
		}
		// 执行DMA映射操作
		ret = nvidia_p2p_dma_map_pages(pdev, page_table, &dma_mapping);
		if (ret) {
			peer_err("nv_dma_map -- error %d while calling nvidia_p2p_dma_map_pages()\n", ret);
			return ret;
		}
		// 检查DMA映射版本是否兼容
		if (!NVIDIA_P2P_DMA_MAPPING_VERSION_COMPATIBLE(dma_mapping)) {
			peer_err("error, incompatible dma mapping version 0x%08x\n",
				 dma_mapping->version);
			nvidia_p2p_dma_unmap_pages(pdev, page_table, dma_mapping);
			return -EINVAL;
		}
		// 获取映射的页数
		nv_mem_context->npages = dma_mapping->entries;
		// 分配sg表
		ret = sg_alloc_table(sg_head, dma_mapping->entries, GFP_KERNEL);
		if (ret) {
			nvidia_p2p_dma_unmap_pages(pdev, page_table, dma_mapping);
			return ret;
		}

		// 设置sg表中的每个散射/聚集项
		nv_mem_context->dma_mapping = dma_mapping;
		// 遍历散射/聚集列表 sg_head->sgl 中的每个散射项
		for_each_sg(sg_head->sgl, sg, nv_mem_context->npages, i) {
			// 将当前散射项 sg 关联到空的页面。这里的空页面表示DMA映射后的数据将被传输到或者从系统内存的哪个位置。
			sg_set_page(sg, NULL, nv_mem_context->page_size, 0);
			// 将 dma_mapping->dma_addresses[i] 的值赋给当前散射项 sg 的 dma_address 属性。这个值表示DMA设备需要访问的物理地址
			sg->dma_address = dma_mapping->dma_addresses[i];
			 将 nv_mem_context->page_size 的值赋给当前散射项 sg 的 dma_length 属性。这个值表示每个散射项所包含的数据长度
			sg->dma_length = nv_mem_context->page_size;
		}
	}
#else
	// 计算需要映射的页数
	// 直接使用页表中的物理地址来设置 SG 表
	nv_mem_context->npages = PAGE_ALIGN(nv_mem_context->mapped_size) >>
						GPU_PAGE_SHIFT;

	// 检查页表中的项数是否与所需页数一致
	if (page_table->entries != nv_mem_context->npages) {
		peer_err("nv_dma_map -- unexpected number of page table entries got=%u, expected=%lu\n",
					page_table->entries,
					nv_mem_context->npages);
		return -EINVAL;
	}

	// 分配sg表
	ret = sg_alloc_table(sg_head, nv_mem_context->npages, GFP_KERNEL);
	if (ret)
		return ret;
	// 设置sg表中的每个散射/聚集项
	for_each_sg(sg_head->sgl, sg, nv_mem_context->npages, i) {
		sg_set_page(sg, NULL, nv_mem_context->page_size, 0);
		sg->dma_address = page_table->pages[i]->physical_address;
		sg->dma_length = nv_mem_context->page_size;
	}
#endif
	// 标记sg表已分配
	nv_mem_context->sg_allocated = 1;
	nv_mem_context->sg_head = *sg_head;
	peer_dbg("allocated sg_head.sgl=%p\n", nv_mem_context->sg_head.sgl);
	// 返回成功映射的页数
	*nmap = nv_mem_context->npages;

	return 0;
}

static int nv_dma_unmap(struct sg_table *sg_head, void *context,
			   struct device  *dma_device)
{
	struct nv_mem_context *nv_mem_context =
		(struct nv_mem_context *)context;
	// 检查 nv_mem_context 是否有效
	if (!nv_mem_context) {
		peer_err("nv_dma_unmap -- invalid nv_mem_context\n");
		return -EINVAL;
	}
	// 检查传入的散射/聚集表是否与存储在 nv_mem_context 中的表匹配
	if (WARN_ON(0 != memcmp(sg_head, &nv_mem_context->sg_head, sizeof(*sg_head))))
		return -EINVAL;
	// 如果当前线程是回调函数的上下文，那么不执行任何操作，直接返回
	if (nv_mem_context->callback_task == current) {
		peer_dbg("no-op in callback context\n");
		goto out;
	}

	peer_dbg("nv_mem_context=%p\n", nv_mem_context);
	// 如果启用了 DMA 映射，执行相应的解除映射操作
#if NV_DMA_MAPPING
	{
		struct pci_dev *pdev = to_pci_dev(dma_device);
		if (nv_mem_context->dma_mapping)
			nvidia_p2p_dma_unmap_pages(pdev, nv_mem_context->page_table,
						   nv_mem_context->dma_mapping);
	}
#endif

out:
	return 0;
}


static void nv_mem_put_pages(struct sg_table *sg_head, void *context)
/*
这个函数的主要功能是释放先前通过 nvidia_p2p_get_pages() 函数映射的页
*/
{

	int ret = 0;
	struct nv_mem_context *nv_mem_context =
		(struct nv_mem_context *) context;
	// 检查 nv_mem_context 是否有效
	if (!nv_mem_context) {
		peer_err("nv_mem_put_pages -- invalid nv_mem_context\n");
		return;
	}
	// 检查散射/聚集表是否与存储在 nv_mem_context 中的表匹配
	if (WARN_ON(0 != memcmp(sg_head, &nv_mem_context->sg_head, sizeof(*sg_head))))
		return;
	// 如果当前线程是回调函数的上下文，那么不执行任何操作，直接返回
	if (nv_mem_context->callback_task == current) {
            	peer_dbg("no-op in callback context\n");
		return;
        }

        peer_dbg("nv_mem_context=%p\n", nv_mem_context);
	// 调用 nvidia_p2p_put_pages() 函数释放先前映射的页
	ret = nvidia_p2p_put_pages(0, 0, nv_mem_context->page_virt_start,
				   nv_mem_context->page_table);

#ifdef _DEBUG_ONLY_
	/* Here we expect an error in real life cases that should be ignored - not printed.
	  * (e.g. concurrent callback with that call)
	*/
	if (ret < 0) {
		printk(KERN_ERR "error %d while calling nvidia_p2p_put_pages, page_table=%p \n",
			ret,  nv_mem_context->page_table);
	}
#endif
	return;
}

static void nv_mem_release(void *context)
// 释放sg表内存和nv_mem_context 结构体内存   

{
	// 将 void 指针类型的 context 转换为 nv_mem_context 指针类型
	struct nv_mem_context *nv_mem_context =
		(struct nv_mem_context *) context;
	// 如果 sg 表已经分配
	if (nv_mem_context->sg_allocated) {
		// 打印调试信息，释放 sg 表
		peer_dbg("freeing sg_head.sgl=%p\n", nv_mem_context->sg_head.sgl);
		sg_free_table(&nv_mem_context->sg_head);
		nv_mem_context->sg_allocated = 0;
	}
	// 释放 nv_mem_context 内存
	kfree(nv_mem_context);
	module_put(THIS_MODULE);
	return;
}

static int nv_mem_get_pages(unsigned long addr,
			  size_t size, int write, int force,
			  struct sg_table *sg_head,
			  void *client_context,
#ifndef PEER_MEM_U64_CORE_CONTEXT
			  void *core_context)
// 获取 GPU 虚拟内存的页面并将其映射到第三方设备
#else
			  u64 core_context)
#endif
{
	int ret;
	// 将 context 强制转换为 nv_mem_context 结构体指针
	struct nv_mem_context *nv_mem_context;

	nv_mem_context = (struct nv_mem_context *)client_context;
	if (!nv_mem_context)
		return -EINVAL;
	// 设置 nv_mem_context 的 core_context 和 page_size 属性
	nv_mem_context->core_context = core_context;
	nv_mem_context->page_size = GPU_PAGE_SIZE;
	// 调用 nvidia_p2p_get_pages 函数，将 GPU 虚拟内存的页面映射到第三方设备
	ret = nvidia_p2p_get_pages(0, 0, nv_mem_context->page_virt_start, nv_mem_context->mapped_size,
			&nv_mem_context->page_table, nv_get_p2p_free_callback, nv_mem_context);
	if (ret < 0) {
		peer_err("nv_mem_get_pages -- error %d while calling nvidia_p2p_get_pages()\n", ret);
		return ret;
	}

	/* No extra access to nv_mem_context->page_table here as we are
	    called not under a lock and may race with inflight invalidate callback on that buffer.
	    Extra handling was delayed to be done under nv_dma_map.
	 */
	return 0;
}


static unsigned long nv_mem_get_page_size(void *context)
// 获取给定上下文中存储的页面大小属性
{
	
	struct nv_mem_context *nv_mem_context =
				(struct nv_mem_context *)context;

	return nv_mem_context->page_size;

}

static struct peer_memory_client_ex nv_mem_client_ex = { .client = {
/*
定义了一个名为 nv_mem_client_ex 的结构体
该结构体用于管理对等内存客户端的各种操作
包括分配内存、获取页面、DMA映射、取消DMA映射、释放页面、获取页面大小和释放内存
*/
	.acquire        = nv_mem_acquire,
	.get_pages  = nv_mem_get_pages,
	.dma_map    = nv_dma_map,
	.dma_unmap  = nv_dma_unmap,
	.put_pages  = nv_mem_put_pages,
	.get_page_size  = nv_mem_get_page_size,
	.release        = nv_mem_release,
}};


static int nv_mem_get_pages_nc(unsigned long addr,
			  size_t size, int write, int force,
			  struct sg_table *sg_head,
			  void *client_context,
#ifndef PEER_MEM_U64_CORE_CONTEXT
			  void *core_context)

#else
			  u64 core_context)
/* 获取页面。它接受地址、大小、写标志、强制标志、散射/聚集表、客户端上下文和核心上下文作为参数
该函数用于获取给定地址范围内的页面，并将页面信息存储在 nv_mem_context 结构体中。
*/
#endif
{
	int ret;
	struct nv_mem_context *nv_mem_context;
	// 打印调试信息，输出地址和大小
	peer_dbg("nv_mem_get_pages_nc -- addr:%lx size:%zu\n", addr, size);
	// 将客户端上下文转换为 nv_mem_context 结构体类型
	nv_mem_context = (struct nv_mem_context *)client_context;
	if (!nv_mem_context)
		return -EINVAL;
	// 断言是否支持持久页面，如果不支持则发出警告
	BUG_ON(!nv_support_persistent_pages());
	// 将核心上下文和页大小设置到 nv_mem_context 结构体中
	nv_mem_context->core_context = core_context;
	nv_mem_context->page_size = GPU_PAGE_SIZE;
	
	// 调用 nvidia_p2p_get_pages 函数获取页面，不设置回调函数
	ret = nvidia_p2p_get_pages(0, 0, nv_mem_context->page_virt_start, nv_mem_context->mapped_size,
			&nv_mem_context->page_table, NULL, NULL);
	if (ret < 0) {
		peer_err("nv_mem_get_pages -- error %d while calling nvidia_p2p_get_pages() with NULL callback\n", ret);
		return ret;
	}

	// 不在此处额外访问 nv_mem_context->page_table，因为我们在没有锁定的情况下调用，并且可能会与正在进行的回调函数竞争。
	// 额外的处理延迟到在 nv_dma_map 中进行。
	return 0;
}

static struct peer_memory_client nv_mem_client_nc = {
/*
定义了一个名为 nv_mem_client_nc 的结构体变量，其中包含了一组函数指针。
这些函数指针指向了对应的函数，用于处理对等内存的不同操作
*/
	.acquire        = nv_mem_acquire,
	.get_pages      = nv_mem_get_pages_nc,
	.dma_map        = nv_dma_map,
	.dma_unmap      = nv_dma_unmap,
	.put_pages      = nv_mem_put_pages,
	.get_page_size  = nv_mem_get_page_size,
	.release        = nv_mem_release,
};


static int __init nv_mem_client_init(void)
/*
初始化函数，用于注册客户端
设置客户端名称和版本号。
注册客户端，包括正常客户端和 NC（非持久化页）客户端。
设置客户端的标志位，允许客户端选择在失效回调期间退出取消映射/放置页面操作。
检查并处理注册过程中的错误情况，包括注册失败时的注销操作。
*/
{
	int status = 0;

	// off by one, to leave space for the trailing '1' which is flagging
	// the new client type
	// 检查驱动名称是否超出最大长度，并将驱动名称复制到客户端名称中
	BUG_ON(strlen(DRV_NAME) > IB_PEER_MEMORY_NAME_MAX-1);
	strcpy(nv_mem_client_ex.client.name, DRV_NAME);

	// [VER_MAX-1]=1 <-- last byte is used as flag
	// [VER_MAX-2]=0 <-- version string terminator
	// 检查驱动版本号是否超出最大长度，并将驱动版本号复制到客户端版本号中
	BUG_ON(strlen(DRV_VERSION) > IB_PEER_MEMORY_VER_MAX-2);
	strcpy(nv_mem_client_ex.client.version, DRV_VERSION);

	// Register as new-style client
	// Needs updated peer_mem patch, but is harmless otherwise
	// 将客户端版本号的最后一个字节设为1，用作标志位
	nv_mem_client_ex.client.version[IB_PEER_MEMORY_VER_MAX-1] = 1;
	// 设置扩展结构的大小
	nv_mem_client_ex.ex_size = sizeof(struct peer_memory_client_ex);

	// PEER_MEM_INVALIDATE_UNMAPS allow clients to opt out of
	// unmap/put_pages during invalidation, i.e. the client tells the
	// infiniband layer that it does not need to call
	// unmap/put_pages in the invalidation callback
	// 设置标志位，允许客户端在失效回调期间选择退出取消映射/放置页面操作
	nv_mem_client_ex.flags = PEER_MEM_INVALIDATE_UNMAPS;
	// 注册为新型客户端
	// 需要更新的 peer_mem 补丁，但否则是无害的
	reg_handle = ib_register_peer_memory_client(&nv_mem_client_ex.client,
						    &mem_invalidate_callback);
	
	if (!reg_handle) {
		// 注册失败时输出错误信息
		peer_err("nv_mem_client_init -- error while registering client\n");
		status = -EINVAL;
		goto out;
	}

	// Register the NC client only if nvidia.ko supports persistent pages
	// 仅在 nvidia.ko 支持持久页时注册 NC 客户端
	if (nv_support_persistent_pages()) {
		// 设置 NC 客户端名称和版本号
		strcpy(nv_mem_client_nc.name, DRV_NAME "_nc");
		strcpy(nv_mem_client_nc.version, DRV_VERSION);
		// 注册 NC 客户端
		reg_handle_nc = ib_register_peer_memory_client(&nv_mem_client_nc, NULL);
		
		if (!reg_handle_nc) {
			peer_err("nv_mem_client_init -- error while registering nc client\n");
			status = -EINVAL;
			goto out;
		}
	}

out:
// 若出现错误，注销已注册的客户端
	if (status) {
		if (reg_handle) {
			ib_unregister_peer_memory_client(reg_handle);
			reg_handle = NULL;
		}

		if (reg_handle_nc) {
			ib_unregister_peer_memory_client(reg_handle_nc);
			reg_handle_nc = NULL;
		}
	}

	return status;
}

static void __exit nv_mem_client_cleanup(void)
{
	// 注销正常客户端
	if (reg_handle)
		ib_unregister_peer_memory_client(reg_handle);
	// 注销 NC 客户端
	if (reg_handle_nc)
		ib_unregister_peer_memory_client(reg_handle_nc);
}

module_init(nv_mem_client_init);
module_exit(nv_mem_client_cleanup);
