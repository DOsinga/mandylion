# mandylion
Generate icons using recurrent neural networks.

icon_rnn.py expects a data directory containing one or more datasets. A dataset should contain a training
subdirectory containing icons to train on. If you don't have one, you can run draw_emoiji.py which will
render the emoiji's as bitmaps. That's only around 1000 images though, which is not very much.

Once you have a dataset in place, train with:

`python icon_rnn.py --dataset=<your set> --mode=train`

When done, you can create a poster of your result with

`python icon_rnn.py --dataset=<your set> --mode=poster`

Or you can pipe generated icons into an image classifier. You need to have inception5h downloaded in the
incpetion5h directory for this to work. You can then run:

`python icon_rnn.py --dataset=<your set> --mode=classify`

The results of these steps are stored in the dataset directory.

