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
............ 작성중 
```

