from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm


templates= [
    "CLASSNAME.",
    "a photomicrograph showing CLASSNAME.",
    "a photomicrograph of CLASSNAME.",
    "an image of CLASSNAME.",
    "an image showing CLASSNAME.",
    "an example of CLASSNAME.",
    "CLASSNAME is shown.",
    "this is CLASSNAME.",
    "there is CLASSNAME.",
    "a histopathological image showing CLASSNAME.",
    "a histopathological image of CLASSNAME.",
    "a histopathological photograph of CLASSNAME.",
    "a histopathological photograph showing CLASSNAME.",
    "shows CLASSNAME.",
    "presence of CLASSNAME.",
    "CLASSNAME is present.",
    "an H&E stained image of CLASSNAME.",
    "an H&E stained image showing CLASSNAME.",
    "an H&E image showing CLASSNAME.",
    "an H&E image of CLASSNAME.",
    "CLASSNAME, H&E stain.",
    "CLASSNAME, H&E."
]
symptoms = ["stromal cell", "immune cell", "squamous cell", "squamous cell carcinoma", "tumor cell", "Neutrophil", "lymphocyte", "well-differentiated tumor cell", "poor-differentiated tumor cell", "endothelial cell"]
threshold = [0.286, 0.275, 0.462, 0.441, 0.372, 0.412, 0.227, 0.318, 0.373, 0.298]

model_cfg = 'conch_ViT-B-16'
checkpoint_path = './CONCH/checkpoints/conch/pytorch_model.bin'
model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path)
tokenizer = get_tokenizer()
_ = model.eval()
model = model.cuda()

# get the embeddings of the combination of the templates and symptoms
template_embeddings = []
for symptom in symptoms:
    text = [template.replace('CLASSNAME', symptom) for template in templates]
    tokens = tokenize(texts=text, tokenizer=tokenizer).cuda()
    with torch.no_grad():
        text_embs = model.encode_text(tokens)
    text_embs = text_embs.mean(dim=0)
    template_embeddings.append(text_embs)
template_embeddings = torch.stack(template_embeddings)


root = ''
tag = ['henan', 'shantou']
# dataset-structure
# root
#  - henan
#    - slide1
#      - patch1.jpeg
#      - ...
#    - slide2
#      - ...
#  - shantou
#    - ...
#  - sysucc
#    - ...

class myDataset(Dataset):
    def __init__(self, root, tag, slide):
        self.root = root
        self.tag = tag
        self.slide = slide
        self.dataset = os.listdir(os.path.join(root, tag, slide))
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.tag, self.slide, self.dataset[idx]))
        img = preprocess(img)
        return img


output_dir = ''
output_feat_dir = ''
# output-structure
# output_dir
#  - henan
#    - slide1
#      - symptom1
#        - patchname_sim{}.jpeg
#        - ...
#      - ...
#    - ...
#  - shantou
#    - ...
#  - sysucc
#    - ...

for t in tag:
    print('Processing {}'.format(t))
    for slide_idx, slide in enumerate(tqdm(os.listdir(os.path.join(root, t)))):
        if not os.path.exists(os.path.join(output_dir, t, slide)):
            os.makedirs(os.path.join(output_dir, t, slide))

        # print slide name and patch number
        print('Processing {}'.format(slide))
        print('Patch number: {}'.format(len(os.listdir(os.path.join(root, t, slide)))))
        dataset = myDataset(root, t, slide)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=12, pin_memory=True)
        image_embeddings = []
        for batch in tqdm(dataloader):
            with torch.no_grad():
                batch = batch.cuda()
                embedding = model.encode_image(batch)
            image_embeddings.append(embedding)
        image_embeddings = torch.cat(image_embeddings)
        # calculate the similarity
        sim = torch.einsum('ij,kj->ik', image_embeddings, template_embeddings)
        # get the patch features that higher than the threshold
        feat_tmp = []
        for idx, symptom in enumerate(symptoms):
            patch_idx = torch.where(sim[:, idx] > threshold[idx])[0]
            for i in patch_idx:
                img = Image.open(os.path.join(root, t, slide, dataset.dataset[i]))
                if not os.path.exists(os.path.join(output_dir, t, slide, symptom)):
                    os.makedirs(os.path.join(output_dir, t, slide, symptom))
                img.save(os.path.join(output_dir, t, slide, symptom, dataset.dataset[i]))
                feat_tmp.append(image_embeddings[i].cpu().numpy())
        feat_tmp = np.array(feat_tmp)
        feat_tmp = torch.from_numpy(feat_tmp)
        # save to pt
        if not os.path.exists(os.path.join(output_feat_dir, t)):
            os.makedirs(os.path.join(output_feat_dir, t))
        torch.save(feat_tmp, os.path.join(output_feat_dir, t, slide+'.pt'))

