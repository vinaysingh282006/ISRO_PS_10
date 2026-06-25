from dataset import TIRSuperResolutionDataset

dataset = TIRSuperResolutionDataset(
    r"C:\Users\Vinay Singh\Desktop\ISRO_PS_10\output\patches"
)

print(len(dataset))

x, y = dataset[0]

print(x.shape, x.dtype)
print(y.shape, y.dtype)