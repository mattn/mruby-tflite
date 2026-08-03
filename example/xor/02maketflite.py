#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.lite as lite

converter = lite.TFLiteConverter.from_keras_model_file("xor_model.h5")
tflite_model = converter.convert()
open("xor_model.tflite", "wb").write(tflite_model)
