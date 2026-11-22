import urllib.request

url = "https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/main/package/office1-1_7-package.tar"

def read_header(offset):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={offset}-{offset+511}"}
    )
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    if len(data) < 512 or data.startswith(b"\x00" * 512):
        return None
    name = data[:100].split(b"\x00")[0].decode("utf-8", errors="ignore")
    # octal size at bytes 124-135
    size_str = data[124:136].split(b"\x00")[0].strip().decode("utf-8", errors="ignore")
    try:
        size = int(size_str, 8)
    except ValueError:
        size = 0
    return name, size

print("Scanning TAR member byte offsets via HTTP Range requests...")
offset = 0
members = []

while True:
    res = read_header(offset)
    if not res:
        break
    name, size = res
    pad = (512 - (size % 512)) % 512
    data_start = offset + 512
    data_end = data_start + size - 1
    next_offset = data_start + size + pad
    print(f"Member: {name:<15} Size: {size:>11} bytes ({size/(1024**3):.3f} GB) Offset: [{data_start} to {data_end}]")
    members.append({
        "name": name,
        "size": size,
        "header_offset": offset,
        "data_start": data_start,
        "data_end": data_end
    })
    offset = next_offset

print(f"Total members cataloged: {len(members)}")
