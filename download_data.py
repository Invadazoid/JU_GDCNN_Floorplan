import requests
from bs4 import BeautifulSoup
import os
import os.path
import zipfile

DATA_PATH = os.path.join(os.getcwd(), 'data')
ROOT_PAGE = "http://mathieu.delalandre.free.fr/projects/sesyd/"
HOMEPAGE = "http://mathieu.delalandre.free.fr/projects/sesyd/index.html"

LINKS_TO_IGNORE = ["../3gT.html", "../../index.html"]
# REQUEST_HEADERS = { 'Host' : "mathieu.delalandre.free.fr",
#                     'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:65.0) Gecko/20100101 Firefox/65.0",
#                     'Accept' : "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
#                     'Accept-Language' : "en-US,en;q=0.5",
#                     'Accept-Encoding' : "gzip, deflate",
#                     'DNT' : '1',
#                     'Connection' : "keep-alive",
#                     'Upgrade-Insecure-Requests' : '1'}


# Host: mathieu.delalandre.free.fr
# User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:65.0) Gecko/20100101 Firefox/65.0
# Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8
# Accept-Language: en-US,en;q=0.5
# Accept-Encoding: gzip, deflate
# Referer: http://mathieu.delalandre.free.fr/projects/sesyd/symbols/symbags.html
# DNT: 1
# Connection: keep-alive
# Upgrade-Insecure-Requests: 1


def unzip_file(path_to_zip_file, directory_to_extract_to):
    try:
        print("         Unzipping: ", path_to_zip_file)
        print("         Unzip Destination: ", directory_to_extract_to)
        zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
        zip_ref.extractall(directory_to_extract_to)
        zip_ref.close()
        print("         Unzipping Done: ", directory_to_extract_to)
    except:
        print("         Unable to unzip...")
        print()


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("        Created directory: ", dir_path)


def download_file(url, download_dest):
    local_filename = os.path.join(download_dest, url.split('/')[-1])
    print("         Destination of download: ", local_filename)
    print("         URL being downloaded: ", url)
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk:
                f.write(chunk)
    print("         Downloaded: ", local_filename)
    directory_to_extract_to = os.path.join(download_dest, "Unzipped")
    make_dir(directory_to_extract_to)
    directory_to_extract_to = os.path.join(directory_to_extract_to, (url.split('/')[-1]).split('.')[0])
    make_dir(directory_to_extract_to)
    unzip_file(local_filename, directory_to_extract_to)


def parse_home_page(site):
    print("===Parsing site: ", site, "====")
    page_response = requests.get(site)
    page_content = BeautifulSoup(page_response.text, 'html.parser')
    links_to_follow = []
    for link in page_content.find_all('a'):
        link = link.get('href')
        print(link)
        if link != LINKS_TO_IGNORE[0] and link != LINKS_TO_IGNORE[1]:
            print("Dataset link found... Adding to list...")
            print("    " + ROOT_PAGE + link)
            print("================")
            links_to_follow.append(ROOT_PAGE + link)
    print()
    print(":::::Following links now:::::")
    print()
    for i, link in enumerate(links_to_follow):
        print("==> Going to link", i+1, "of", len(links_to_follow), ":  ", link)
        follow_page(link)


def follow_page(site):
    folder = site.split('/')[-2]
    download_dest = os.path.join(DATA_PATH, folder)
    make_dir(download_dest)
    subfolder = (site.split('/')[-1]).split('.')[0]
    download_dest = os.path.join(download_dest, subfolder)
    make_dir(download_dest)
    page_response = requests.get(site)
    page_content = BeautifulSoup(page_response.text, 'html.parser')
    for link in page_content.find_all('a'):
        link = link.get('href')
        if ".zip" in link:
            link = ROOT_PAGE + folder + '/' + link
            print("    Download link found: ", link)
            download_file(link, download_dest)

if __name__ == '__main__':
    make_dir(DATA_PATH)
    parse_home_page(HOMEPAGE)
