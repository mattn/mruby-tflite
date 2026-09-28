# Examples

Build an mruby that has this gem in it:

```
cd example
make
```

That clones mruby, builds it with `build_config.rb`, and leaves the binary at
`example/mruby`. `make MRUBY_SRC=~/mruby` uses an mruby checkout you already
have instead of cloning one, and `make MRUBY_VERSION=3.3.0` picks another
branch or tag.

Then run any of the scripts below from the directory it lives in:

```
cd xor
../mruby xor.rb
```

## xor

The smallest useful model there is: two inputs, one output.

```
$ ../mruby xor.rb
0 ^ 0 = 0
1 ^ 0 = 1
0 ^ 1 = 1
1 ^ 1 = 0
```

`makexor.py` trains it and `maketflite.py` converts it to `xor_model.tflite`.

## fizzbuzz

FizzBuzz decided by a network instead of by `%`. The number goes in as 7 bits
and the answer comes out as 4 scores.

```
$ ../mruby fizzbuzz.rb
1
2
Fizz
...
```

`make.py` trains and converts the model.

## inspect

Print the tensor layout of any `.tflite` file, which is the first thing you
want to know when writing code against a model you did not train yourself.

```
$ ../mruby inspect.rb ../xor/xor_model.tflite
../xor/xor_model.tflite
  1 input(s)
    dense_1_input: float32[1, 2] (8 bytes)
  1 output(s)
    activation_2/Sigmoid: float32[1, 1] (4 bytes)
```

## benchmark

Invokes per second for a few thread counts, via `InterpreterOptions`.

```
$ ../mruby benchmark.rb ../mobilenet/mobilenet_v1_1.0_224_quant.tflite
1 thread(s): 60 invokes in 1241ms (48/sec)
2 thread(s): 80 invokes in 1086ms (74/sec)
4 thread(s): 100 invokes in 1159ms (86/sec)
```

## mobilenet

Image classification with MobileNet V1, i.e. a model big enough to be
interesting. `make` downloads the model and its labels, and any converter that
writes a binary PPM prepares the image.

```
$ make
$ ffmpeg -i hopper.jpg -vf scale=224:224 -pix_fmt rgb24 hopper.ppm
$ ../mruby classify.rb hopper.ppm
hopper.ppm
  89.0% military uniform
  2.4% Windsor tie
  1.2% mortarboard
  0.8% bow tie
  0.8% bulletproof vest
```
