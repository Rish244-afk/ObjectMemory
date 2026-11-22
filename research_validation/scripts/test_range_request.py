import urllib.request

url = "https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/main/package/office1-1_7-package.tar"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-1023"})

try:
    with urllib.request.urlopen(req) as resp:
        print("Status:", resp.status)
        print("Content-Range:", resp.headers.get("Content-Range"))
        print("Accept-Ranges:", resp.headers.get("Accept-Ranges"))
        data = resp.read()
        print("Read bytes:", len(data))
        tar_header = data[:512]
        file_name = tar_header[:100].split(b"\x00")[0].decode("utf-8", errors="ignore")
        print("First TAR member name:", file_name)
except Exception as e:
    print("Error:", e)
