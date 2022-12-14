# dataLoader

링크 분실.... 찾으면 업데이트 하겠습니다...

```python
class BaseDataset(Dataset):
    def __init__(self,img_paths:str,labels = None,cache = None,use_caching = False, transform = None) -> None:
        super().__init__()
        self.img_paths = img_paths
        self.transform = transform
        self.cache = cache
        self.labels = labels
        self.use_caching = use_caching
    def __len__(self)->int:
        return len(self.img_paths)
        
class CacheDataset(BaseDataset):
    def __getitem__(self,index):
        path = self.img_paths[index]
        image  = self.imread_cache(path)
        if self.transform is not None:
            # image = self.transform(image) #torchvision version
            image = self.transform(image=image)['image'] #Albumentation ver
        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image        
        
        
    def imread_cache(self,path):
        img = self.cache.get(path, None)  #cache에 들어있다면, 가지고 오지만 없으면 None을 return
        if img is None:                   #cache값이 None이라면, 
            img = cv2.imread(path)        #imread로 읽고 , cv2가 PIL보다 빠르고 tool도 더 많다.
            # img = cv2.resize(img, (self.imsize, self.imsize))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #RGB로 변환 
            self.cache[path] = img
        else:
            pass
        return img
```

