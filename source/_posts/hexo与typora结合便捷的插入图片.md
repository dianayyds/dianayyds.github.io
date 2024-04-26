---
title: hexo+typora插入图片
typora-root-url: hexo与typora结合便捷的插入图片
tags:
  - hexo
  - typora
date: 2024-04-26 13:04:00
summary: 解决hexo插入图片无法显示的问题
categories: hexo
---

### 解决hexo插入图片无法显示的问题

在md中插入图片十分简单，格式如下:

``` markdown
![显示在图片下方的名字](图片地址)
```

但是部署的时候，并不是完全把整个文件部署上去，导致了文件的路径很混乱，我查阅了网上许多的资料，下面是我的解决方法：



## 一

在hexo根目录的config里面找到post_asset_folder 设置为true，以后新建文章的时候使用hexo new "文章名字",会在同文件夹下自动创建和你的文章名字相同的资源文件夹。

*注意如果按照网上设置了marked的两个属性，一定全部删掉，属于误人子弟！*

## 二

在hexo根目录下运行:

```bash
npm install https://github.com/CodeFalling/hexo-asset-image --save
```

解析图片的插件



## 三

进入typora中点击文件->偏好设置，配置如下:

![](/image-20240426132030054.png)

使得你在粘贴进typora的时候，自动保存图片并指定图片路径

## 四

在hexo根目录的scaffolds->post中添加设置：

```yaml
typora-root-url: {{ title }}
```

以后每次新建博客都会在front-matter中出现 typora-root-url: "标题名字",粘贴图片的时候就会显示相对路径了



#### 这样就完成了



可以直接把任意图片粘贴进typora中，部署和预览都不需要其他操作
