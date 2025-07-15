import os
import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import quote
from PIL import Image
import io
# 爬取狗的图片
class DogImageCrawler:
    def __init__(self):
        # 初始化 headers
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }
        
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]
        
        # 扩展狗品种列表
        self.dog_breeds = {
            '德国牧羊犬': 'german shepherd',
            '金毛寻回犬': 'golden retriever',
            '法国斗牛犬': 'french bulldog',
            '贵宾犬': 'poodle',
            '柴犬': 'shiba inu',
            '边境牧羊犬': 'border collie',
            '比熊犬': 'bichon frise',
            '柯基': 'welsh corgi',
            '萨摩耶': 'samoyed',
            '杜宾犬': 'doberman',
            '大丹犬': 'great dane',
            '圣伯纳犬': 'st bernard',
            '约克夏': 'yorkshire terrier'
        }
        
        # 修改搜索引擎headers，添加更多选项
        self.search_headers = {
            'sogou': {
                'Host': 'pic.sogou.com',
                'Referer': 'https://pic.sogou.com/pics',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            },
            '360': {
                'Host': 'image.so.com',
                'Referer': 'https://image.so.com',
                'Accept': 'application/json, text/plain, */*'
            },
            'sina': {
                'Host': 'pic.sina.com.cn',
                'Referer': 'https://pic.sina.com.cn',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            }
        }

    def create_folders(self):
        """创建数据文件夹结构"""
        # 创建训练、验证和测试集文件夹
        for split in ['train', 'val', 'test']:
            for breed in self.dog_breeds.keys():
                path = os.path.join('data', split, breed)
                os.makedirs(path, exist_ok=True)
    
    def is_valid_image(self, content):
        """验证图片是否有效且符合要求"""
        try:
            img = Image.open(io.BytesIO(content))
            # 转换为RGB模式
            img = img.convert('RGB')
            # 检查图片尺寸
            width, height = img.size
            if width < 200 or height < 200:
                return False
            return True
        except:
            return False

    def download_image(self, url, save_path, max_retries=3):
        """下载并保存图片，支持重试"""
        for attempt in range(max_retries):
            try:
                headers = self.headers.copy()
                headers['User-Agent'] = random.choice(self.user_agents)
                
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200 and self.is_valid_image(response.content):
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    return True
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"下载失败 {url}: {str(e)}")
                time.sleep(random.uniform(1, 3))
        return False

    def search_baidu_images(self, keyword, page=0):
        """搜索百度图片"""
        base_url = f"https://image.baidu.com/search/flip?tn=baiduimage&word={quote(keyword)}&pn={page * 30}"
        try:
            response = requests.get(base_url, headers=self.headers)  # 修复未关闭的括号
            # 使用字符串查找的方式提取图片URL
            img_urls = []
            content = response.text
            
            # 查找所有图片URL
            start_index = 0
            while True:
                # 查找图片URL的起始位置
                pos = content.find('"objURL":"', start_index)
                if pos == -1:
                    break
                    
                # 提取URL
                start = pos + 10
                end = content.find('",', start)
                if end == -1:
                    break
                    
                img_url = content[start:end]
                if img_url.startswith('http'):
                    img_urls.append(img_url)
                    
                start_index = end + 1
            
            return img_urls
            
        except Exception as e:
            print(f"搜索失败: {str(e)}")
            return []

    def search_sogou_images(self, keyword, page=1):
        """搜索搜狗图片 - 改进版"""
        base_url = f"https://pic.sogou.com/napi/pc/searchList?mode=1&start={page * 48}&xml_len=48&query={quote(keyword)}"
        headers = {**self.headers, **self.search_headers['sogou']}
        try:
            response = requests.get(base_url, headers=headers)
            data = response.json()
            items = data.get('data', {}).get('items', [])
            return [item.get('picUrl') for item in items if item.get('picUrl')]
        except Exception as e:
            print(f"搜狗搜索失败: {str(e)}")
            return []

    def search_360_images(self, keyword, page=1):
        """搜索360图片"""
        base_url = f"https://image.so.com/j?q={quote(keyword)}&pn={page * 30}"
        headers = {**self.headers, **self.search_headers['360']}
        try:
            response = requests.get(base_url, headers=headers)
            data = response.json()
            return [item['img'] for item in data.get('list', [])]
        except Exception as e:
            print(f"360搜索失败: {str(e)}")
            return []

    def search_sina_images(self, keyword, page=1):
        """搜索新浪图片"""
        base_url = f"https://pic.sina.com.cn/search/index.php?k={quote(keyword)}&p={page}"
        headers = {**self.headers, **self.search_headers['sina']}
        try:
            response = requests.get(base_url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            img_elements = soup.find_all('img', class_='img')
            return [img.get('src') for img in img_elements if img.get('src')]
        except Exception as e:
            print(f"新浪搜索失败: {str(e)}")
            return []

    def distribute_images(self, breed, total_images):
        """按比例分配图片到训练、验证和测试集"""
        # 分配比例：70% 训练，15% 验证，15% 测试
        train_size = int(total_images * 0.7)
        val_size = int(total_images * 0.15)
        test_size = total_images - train_size - val_size
        
        return {
            'train': train_size,
            'val': val_size,
            'test': test_size
        }

    def crawl_breed(self, breed, total_images=100):
        """改进的爬取方法，支持多个来源和错误处理"""
        print(f"开始下载 {breed} 的图片...")
        
        distribution = self.distribute_images(breed, total_images)
        downloaded = {split: 0 for split in ['train', 'val', 'test']}
        
        # 扩展搜索源列表
        search_sources = [
            (self.search_baidu_images, "百度"),
            (self.search_sogou_images, "搜狗"),
            (self.search_360_images, "360"),
            (self.search_sina_images, "新浪")
        ]
        
        # 记录已下载的URL，避免重复
        downloaded_urls = set()
        
        page = 0
        consecutive_failures = 0
        current_source_index = 0
        
        while sum(downloaded.values()) < total_images and page < 50:
            if consecutive_failures >= 5:
                print(f"连续5次失败，切换到下一个搜索源")
                consecutive_failures = 0
                current_source_index = (current_source_index + 1) % len(search_sources)
                continue
            
            search_func, source_name = search_sources[current_source_index]
            print(f"从{source_name}搜索 {breed}, 第{page+1}页")
            
            english_name = self.dog_breeds.get(breed)
            keywords = [
                f"{english_name} dog breed",
                f"{english_name} dog",
                f"{breed} 狗",
                f"{breed} 宠物狗"
            ]
            
            for keyword in keywords:
                try:
                    urls = search_func(keyword, page)
                    new_urls = [url for url in urls if url not in downloaded_urls]
                    
                    if not new_urls:
                        consecutive_failures += 1
                        continue
                    
                    for url in new_urls:
                        for split in ['train', 'val', 'test']:
                            if downloaded[split] < distribution[split]:
                                save_path = os.path.join('data', split, breed, 
                                                       f"{breed}_{source_name}_{downloaded[split]+1}.jpg")
                                
                                if self.download_image(url, save_path):
                                    downloaded_urls.add(url)
                                    downloaded[split] += 1
                                    print(f"已下载: {sum(downloaded.values())}/{total_images} - 来源: {source_name}")
                                    consecutive_failures = 0
                                    
                                    if sum(downloaded.values()) >= total_images:
                                        print(f"\n完成 {breed} 的下载:")
                                        print(f"总计: {sum(downloaded.values())} 张图片")
                                        print(f"分布: 训练集 {downloaded['train']}, "
                                              f"验证集 {downloaded['val']}, "
                                              f"测试集 {downloaded['test']}")
                                        return
                                
                                break
                    
                    time.sleep(random.uniform(0.5, 1))
                
                except Exception as e:
                    print(f"搜索出错 ({source_name}): {str(e)}")
                    consecutive_failures += 1
                
                time.sleep(random.uniform(1, 2))
            
            page += 1
            if page % 5 == 0:  # 每5页切换一次搜索源
                current_source_index = (current_source_index + 1) % len(search_sources)
                page = 0  # 重置页码
            
            time.sleep(random.uniform(2, 3))

    def start_crawling(self, images_per_breed=100):
        """开始爬取所有品种的图片"""
        self.create_folders()
        
        for breed in self.dog_breeds.keys():
            self.crawl_breed(breed, images_per_breed)
            time.sleep(random.uniform(2, 3))

if __name__ == "__main__":
    crawler = DogImageCrawler()
    crawler.start_crawling(100)  # 每个品种下载100张图片 