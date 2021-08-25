# Data Loader
import torchvision.datasets
from SpykeTorch import utils
from SpykeTorch import functional as sf
from torchvision import transforms
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader


class Transform_MNIST:
	def __init__(self, timesteps = 15):
		self.kernels = [ utils.DoGKernel(7,1,2),
                         utils.DoGKernel(7,2,1)]
        
		self.filter = utils.Filter(self.kernels, padding = 3, thresholds = 50)
		self.in2lan = utils.Intensity2Latency(timesteps)
        
        
        
	def __call__(self, image):
		image = transforms.ToTensor()(image) * 255
		image.unsqueeze_(0)
		image = self.filter(image)        
		image = sf.local_normalization(image, 8)
		temporal_image = self.in2lan(image)
		return temporal_image.sign().byte()

class Transform_Caltech:
	def __init__(self,
                 pooling_size,
                 pooling_stride,
                 lateral_inhibition = None,
                 timesteps = 15,
			     feature_wise_inhibition=True):
        
		self.transform = transforms.Compose((
            transforms.Resize((120, 120)),
            transforms.Grayscale(),
            transforms.ToTensor()))
        
		self.kernels = [utils.GaborKernel(5, 45+22.5),
			            utils.GaborKernel(5, 90+22.5),
			            utils.GaborKernel(5, 135+22.5),
			            utils.GaborKernel(5, 180+22.5)]
        
		self.filter = utils.Filter(self.kernels, use_abs = True)
		self.pooling_size = pooling_size
		self.pooling_stride = pooling_stride
		self.lateral_inhibition = lateral_inhibition
		self.in2lan = utils.Intensity2Latency(timesteps)
		self.feature_wise_inhibition = feature_wise_inhibition
        
	def __call__(self, image):
		image = self.transform(image)

		image.unsqueeze_(0)
		image = self.filter(image)
		image = sf.pooling(image, self.pooling_size, self.pooling_stride, padding=self.pooling_size//2)
		if self.lateral_inhibition is not None:
 			image = self.lateral_inhibition(image)
# 		image = sf.local_normalization(image, 8)
		temporal_image = self.in2lan(image)
		temporal_image = sf.pointwise_inhibition(temporal_image)
		return temporal_image.sign().byte()



def Loader(dataName):
    if dataName == 'MNIST':
        inp = Transform_MNIST()
        # lateral_inhibition = utils.LateralIntencityInhibition([0.15, 0.12, 0.1, 0.07, 0.05])
        # inp = Transform_Caltech(7, 6, lateral_inhibition)
        trainsetfolder = utils.CacheDataset(torchvision.datasets.MNIST(root="data", train=True, download=True, transform = inp))
        testsetfolder = utils.CacheDataset(torchvision.datasets.MNIST(root="data", train=False, download=True, transform = inp))
        trainset = DataLoader(trainsetfolder, batch_size=len(trainsetfolder), shuffle=False)
        testset = DataLoader(testsetfolder, batch_size=len(testsetfolder), shuffle=False)
        return trainset, testset
    
    elif dataName == 'Caltech':
        lateral_inhibition = utils.LateralIntencityInhibition([0.15, 0.12, 0.1, 0.07, 0.05])
        inp = Transform_Caltech(7, 6, lateral_inhibition)
        trainsetfolder = utils.CacheDataset(ImageFolder("data/caltech101/train", inp))
        testsetfolder = utils.CacheDataset(ImageFolder("data/caltech101/test", inp))
        trainset = DataLoader(trainsetfolder, batch_size=len(trainsetfolder), shuffle=True)
        testset = DataLoader(testsetfolder, batch_size=len(testsetfolder), shuffle=True)
        return trainset, testset
    
    assert (dataName == 'MNIST' or dataName == 'Caltech')