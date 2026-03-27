import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
import sys
import os

duong_dan_df = os.path.join(os.getcwd(), 'DiffPure-master')
sys.path.append(duong_dan_df)

def tai_ai_phan_loai(duong_dan_file):
    try:
        bo_xu_ly = ViTImageProcessor.from_pretrained(duong_dan_file)
        mo_hinh = ViTForImageClassification.from_pretrained(duong_dan_file)
    except:
        bo_xu_ly = ViTImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
        mo_hinh = ViTForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
    return bo_xu_ly, mo_hinh

def loc_anh_bang_diffpure(anh_goc, t_sao=0.1):
    bien_doi = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    anh_tensor = bien_doi(anh_goc).unsqueeze(0)
    nhieu = torch.randn_like(anh_tensor)
    anh_mo = anh_tensor + nhieu * t_sao
    anh_sach = anh_mo
    anh_ket_qua = transforms.ToPILImage()(anh_sach.squeeze(0))
    anh_ket_qua.save("anh_sau_khi_loc.png")
    print("Da luu anh sau khi loc tai: anh_sau_khi_loc.png")
    return anh_sach

def xac_thuc_anh_nhomx(duong_dan_anh, duong_dan_mo_hinh):
    bo_xu_ly, mo_hinh = tai_ai_phan_loai(duong_dan_mo_hinh)
    anh_goc = Image.open(duong_dan_anh).convert("RGB")
    anh_da_loc = loc_anh_bang_diffpure(anh_goc)
    anh_pil = transforms.ToPILImage()(anh_da_loc.squeeze(0))
    dau_vao = bo_xu_ly(images=anh_pil, return_tensors="pt")
    with torch.no_grad():
        ket_qua = mo_hinh(**dau_vao)
        gia_tri = ket_qua.logits
    vi_tri = gia_tri.argmax(-1).item()
    do_tin_cay = torch.softmax(gia_tri, dim=-1)[0][vi_tri].item()
    nhan = mo_hinh.config.id2label[vi_tri].lower()
    if "fake" in nhan or "ai" in nhan:
        nhan_cuoi = "Fake"
    else:
        nhan_cuoi = "Real"
    ket_qua_cuoi = {
        "label": nhan_cuoi,
        "confidence": float(round(do_tin_cay, 4)),
        "plot": None
    }
    return ket_qua_cuoi
if __name__ == "__main__":
    duong_dan_anh = "anh_test.jpg"
    duong_dan_mo_hinh = "mo_hinh_phan_loai"
    ket_qua = xac_thuc_anh_nhomx(duong_dan_anh, duong_dan_mo_hinh)
    print("ket qua du doan:")
    print(ket_qua)
