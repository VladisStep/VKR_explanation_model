from fastai.vision import *
from fastai.vision.transform import *

# https://www.kaggle.com/schorsi/classification-of-alzheimers-95-totw/
if __name__ == '__main__':
    PATH = Path('../Alzheimer_s Dataset/')

    transform = get_transforms(max_rotate=7.5,
                               max_zoom=1.15,
                               max_lighting=0.15,
                               max_warp=0.15,
                               p_affine=0.8, p_lighting=0.8,
                               xtra_tfms=[
                                   pad(mode='zeros'),
                                   symmetric_warp(magnitude=(-0.1, 0.1)),
                                   cutout(n_holes=(1, 3), length=(5, 5))
                               ])

    data = ImageDataBunch.from_folder(PATH, train="train/",
                                      test="test/",
                                      valid_pct=.4,
                                      ds_tfms=transform,
                                      size=112, bs=64,
                                      ).normalize(imagenet_stats)

    data.show_batch(rows=3, figsize=(10, 10))

    Category.__eq__ = lambda self, that: self.data == that.data
    Category.__hash__ = lambda self: hash(self.obj)
    Counter(data.train_ds.y)

    learn = cnn_learner(data, models.vgg16_bn, metrics=[FBeta(average='weighted'), accuracy], wd=1e-1,
                        callback_fns=ShowGraph)

    learn.fit_one_cycle(1)

    learn.export()

    Model_Path = Path('./Alzheimer-stage-classifier-model/')
    learn.model_dir = Model_Path
    learn.save('checkpoint-1')

    learn.recorder.plot_losses()
