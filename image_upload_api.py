import http.server
import socketserver
import requests
from urllib.parse import urlparse

# 图片文件路径
# image_path = 'path/to/image.png'


# 定义请求处理器类
class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # 处理GET请求
        try:
            bits = urlparse(self.path)
            if bits.path != '/image':
                # 其他路径返回404错误
                self.send_error("path error!")
            query = bits.query
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            print(image_path)
            print(query)
            with open(image_path + query, 'rb') as f:
                self.wfile.write(f.read())
        except:
            # 其他路径返回404错误
            self.send_error("error!")

# 启动Web服务器
def web_start(image_path):
    PORT = 8001
    handler = MyHandler
    httpd = socketserver.TCPServer(("", PORT), handler)
    print("serving at port", PORT)
    httpd.serve_forever()


if __name__ == "__main__":
    image_path = "/home/ubuntu/stable_diffusion/images/"
    web_start(image_path)