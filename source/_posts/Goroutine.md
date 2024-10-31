---
title: Goroutine
typora-root-url: Goroutine
date: 2024-09-16 20:25:48
summary: Goroutine 是 Go 语言的并发编程模型，它是一种轻量级的线程，由 Go 运行时管理，我们也可以称之为协程。
categories: Golang必学概念
---

### 优点

- **轻量级**：Goroutine 的栈空间初始大小只有 2KB，可以动态扩容，最大可达 1GB
- **快速启动**：Goroutine 的启动时间只有 1~2us
- **高效调度**：Goroutine 的调度器采用 M:N 模型，可以将 M 个 Goroutine 映射到 N 个 OS 线程上，实现高效调度
- **通信简单**：Goroutine 之间通过 Channel 进行通信，实现数据共享
- **无锁**：Goroutine 之间通过 Channel 进行通信，无需加锁
- **高并发**：Goroutine 可以轻松创建数十万个，实现高并发
- **高性能**：Goroutine 的调度器采用抢占式调度，实现高性能



### 创建Goroutine

由于 Goroutine 是 Golang 非常重视的基本功能，因此在 Golang 中创建异步 Goroutine 非常简单，只需要在函数调用前加上 `go` 关键字即可，比绝大部分的编程语言都要简单。



```go
package main

import (
    "fmt"
    "time"
)

func main() {
    go func() {
        for {
            fmt.Println("running...")
            time.Sleep(time.Second)
        }
    }()

    time.Sleep(5 * time.Second)
}
```

使用 `go` 加上任意 `func` 即可创建一个 Goroutine，Goroutine 会在后台执行，不会阻塞主线程。



### 停止 Goroutine

- **运行结束**：Goroutine 会在函数运行结束后自动结束
- **超时结束**：通过 `context.WithTimeout()` 或 `context.WithDeadline()` 可以设置 Goroutine 的超时时间
- **手动结束**：通过 `context.WithCancel()` 可以手动结束 Goroutine
- **通道结束**：通过 Channel 通信，可以结束 Goroutine



### Goroutine和Channel

我们知道，无论是在线程还是协程，在运行的时候都会遇到共享数据或传递数据的情况，在 Golang 中，我们可以通过 Channel 来实现 Goroutine 之间的通信。





```go
package main

import (
    "fmt"
    "time"
)

func main() {
    ch := make(chan int)
go func() {
    for {
        select {
        case <-ch:
            fmt.Println("exit")
            return
        default:
            fmt.Println("running...")
            time.Sleep(time.Second)
        }
    }
}()

time.Sleep(5 * time.Second)
ch <- 1
}
```


在上面的例子中，我们创建了一个 Channel `ch`，在主线程中向 `ch` 中发送了一个数据，Goroutine 中通过 `select` 语句监听 `ch`，当 `ch` 中有数据时，Goroutine 会退出。

协程之间通过 Channel 通信的例子：

```go
package main

import (
	"fmt"
	"time"
)

func main() {
	ch := make(chan string)

	go sendData(ch)
	go getData(ch)

	time.Sleep(time.Second)
}

func sendData(ch chan string) {
	ch <- "ycx"
	ch <- "hexo"
}

func getData(ch chan string) {
	var input string
	for {
		input = <-ch
		fmt.Printf("%s ", input)
	}
}

// 结果: ycx hexo
```

