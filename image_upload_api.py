import http.server
import socketserver
import requests

# 图片文件路径
# image_path = 'path/to/image.png'


# 定义请求处理器类
class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # 处理GET请求
        if self.path == '/image.png':
            # 如果请求路径是'/image.png'，则返回该图片文件
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            with open(image_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            # 其他路径返回404错误
            self.send_error(404)

# 启动Web服务器
def web_start(image_path):
    PORT = 8001
    handler = MyHandler
    httpd = socketserver.TCPServer(("", PORT), handler)
    print("serving at port", PORT)
    httpd.serve_forever()


if __name__ == "__main__":
    image_path = "/home/ubuntu/stable_diffusion/images/image.png"
    web_start(image_path)