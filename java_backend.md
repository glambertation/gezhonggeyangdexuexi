## FILEIO

https://www.jianshu.com/p/c752767cff22

https://blog.csdn.net/amo_te_ama_me/article/details/40587997

三种read，read(),read(byte[]), read(byte[] b, int off, int len)

```
FileInputStream input=new FileInputStream(oldAddress);
FileOutputStream output=new FileOutputStream(newAddress);
byte[] buffer=new byte[500];
int in=input.read(buffer);
while(in!=-1){
  output.write(buffer,0,in);
  in=input.read(buffer);
        }
```

read()是一个字节一个字节读，奇慢无比，而且我不知道为啥它还阻塞了，fileio为啥能阻塞

## webservice

#### SOAP

Web service建好以后，你或者其他人就会去调用它。简单对象访问协议(SOAP)提供了标准的RPC方法来调用Web service。实际上，SOAP在这里有点用词不当：它意味着下面的Web service是以对象的方式表示的，但事实并不一定如此：你完全可以把你的Web service写成一系列的C函数，并仍然使用SOAP进行调用。SOAP规范定义了SOAP消息的格式，以及怎样通过HTTP协议来使用SOAP。SOAP也是基于XML（标准通用标记语言下的一个子集）和XSD的，XML是SOAP的数据编码方式。

#### WSDL

你会怎样向别人介绍你的Web service有什么功能，以及每个函数调用时的参数呢？你可能会自己写一套文档，你甚至可能会口头上告诉需要使用你的Web service的人。这些非正式的方法至少都有一个严重的问题：当程序员坐到电脑前，想要使用你的Web service的时候，他们的工具(如Visual Studio)无法给他们提供任何帮助，因为这些工具根本就不了解你的Web service。


## servlet
request.getServletPath()，request.getContextPath()

https://blog.csdn.net/qq_27770257/article/details/79438987

通过观察打印结果，我们可以总结： 
1. getServletPath():获取能够与“url-pattern”中匹配的路径，注意是完全匹配的部分，*的部分不包括。 
2. getPageInfo():与getServletPath()获取的路径互补，能够得到的是“url-pattern”中*d的路径部分 
3. getContextPath():获取项目的根路径 
4. getRequestURI:获取根路径到地址结尾 
5. getRequestURL:获取请求的地址链接（浏览器中输入的地址） 
6. getServletContext().getRealPath(“/”):获取“/”在机器中的实际地址 
7. getScheme():获取的是使用的协议(http 或https) 
8. getProtocol():获取的是协议的名称(HTTP/1.11) 
9. getServerName():获取的是域名(xxx.com) 
10. getLocalName:获取到的是IP




## websocket

https://blog.csdn.net/yalishadaa/article/details/72235529

初次接触 WebSocket 的人，都会问同样的问题：我们已经有了 HTTP 协议，为什么还需要另一个协议？它能带来什么好处？

答案很简单，因为 HTTP 协议有一个缺陷：通信只能由客户端发起。

举例来说，我们想了解今天的天气，只能是客户端向服务器发出请求，服务器返回查询结果。HTTP 协议做不到服务器主动向客户端推送信息。

WebSocket 协议在2008年诞生，2011年成为国际标准。所有浏览器都已经支持了。

它的最大特点就是，服务器可以主动向客户端推送信息，客户端也可以主动向服务器发送信息，是真正的双向平等对话，属于服务器推送技术的一种。

其他特点包括：

（1）建立在 TCP 协议之上，服务器端的实现比较容易。

（2）与 HTTP 协议有着良好的兼容性。默认端口也是80和443，并且握手阶段采用 HTTP 协议，因此不容易屏蔽，能通过各种 HTTP 代理服务器。

（3）数据格式比较轻量，性能开销小，通信高效。

（4）可以发送文本，也可以发送二进制数据。

（5）没有同源限制，客户端可以与任意服务器通信。

（6）协议标识符是ws（如果加密，则为wss），服务器网址就是 URL。

ws://example.com:80/some/path
