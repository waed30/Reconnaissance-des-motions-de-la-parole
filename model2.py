from google.colab import drive
drive.mount('/content/drive')
from fastai.vision import *
path=Path('/content/drive/My Drive/colab/datacnn')
tfms = get_transforms(do_flip=False,flip_vert=False, max_lighting=0.1, max_zoom=1.05, max_warp=0.,max_rotate=5)
np.random.seed(42)
data.classes
data.show_batch(rows=3, figsize=(7,8))
learn = cnn_learner(data, models.resnet34, metrics=accuracy)
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-5,1e-4))
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-5,1e-4))
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=tfms, size=288, num_workers=4).normalize(imagenet_stats)
data.batch_size=25
learn.data=data
data.train_ds[0][0].shape
data.classes
data.show_batch(rows=3, figsize=(7,8))
learn.freeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-4))
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-5,1e-4))
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-4))
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-5))
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, slice(1e-5,1e-4))
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
