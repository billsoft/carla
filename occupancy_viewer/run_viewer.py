#!/usr/bin/env python
# -*- coding: utf-8 -*-

import http.server
import socketserver
import json
import os
from pathlib import Path
import urllib.parse
import mimetypes

# 配置
PORT = 8000
VIEWER_DIR = Path(__file__).parent.absolute()
DATA_DIR = Path(r"d:\code\carla\dataset_output\occupancy")

class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # 解析 URL
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        # API: 获取文件列表
        if path == '/api/list':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            files = []
            if DATA_DIR.exists():
                # 获取所有 .npz 文件
                files = [f.name for f in DATA_DIR.glob('*.npz')]
                files.sort()
            
            self.wfile.write(json.dumps(files).encode('utf-8'))
            return

        # API: 获取数据文件
        if path.startswith('/data/'):
            filename = path.replace('/data/', '')
            file_path = DATA_DIR / filename
            
            if file_path.exists() and file_path.is_file():
                try:
                    # 获取文件大小 (使用 pathlib 的 stat 方法，更稳健)
                    file_size = file_path.stat().st_size
                    
                    self.send_response(200)
                    # 设置 MIME 类型
                    self.send_header('Content-type', 'application/octet-stream')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header("Content-Length", str(file_size))
                    self.end_headers()
                    
                    # 发送文件内容
                    with open(file_path, 'rb') as f:
                        self.copyfile(f, self.wfile)
                except Exception as e:
                    print(f"Error serving file {file_path}: {e}")
                    # 如果 header 还没发送，发送 500
                    try:
                        self.send_error(500, f"Internal Server Error: {e}")
                    except:
                        pass
                return
            else:
                self.send_error(404, "File not found")
                return

        # 默认行为: 静态文件服务
        # 如果请求根目录，重定向到 index.html
        if path == '/' or path == '/occupancy_viewer/':
            self.path = '/index.html'
        
        # 移除 /occupancy_viewer 前缀 (如果存在)
        if self.path.startswith('/occupancy_viewer/'):
            self.path = self.path.replace('/occupancy_viewer/', '/')

        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    # 允许跨域
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        http.server.SimpleHTTPRequestHandler.end_headers(self)

def run_server():
    # 切换工作目录到 viewer 目录
    os.chdir(VIEWER_DIR)
    
    print(f"="*60)
    print(f"Occupancy Viewer Server")
    print(f"="*60)
    print(f"Viewer Directory: {VIEWER_DIR}")
    print(f"Data Directory:   {DATA_DIR}")
    print(f"URL:              http://localhost:{PORT}/")
    print(f"="*60)

    if not DATA_DIR.exists():
        print(f"WARNING: Data directory does not exist: {DATA_DIR}")

    with socketserver.TCPServer(("", PORT), ViewerHandler) as httpd:
        print(f"Serving at port {PORT}...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == '__main__':
    run_server()
