MODEL = mobilenet_v1_1.0_224_quant.tflite
LABELS = labels_mobilenet_quant_v1_224.txt
ARCHIVE = mobilenet_v1_1.0_224_quant_and_labels.zip
URL = https://storage.googleapis.com/download.tensorflow.org/models/tflite/$(ARCHIVE)

all: $(MODEL) $(LABELS)

$(MODEL) $(LABELS): $(ARCHIVE)
	unzip -o -j $(ARCHIVE) $(MODEL) $(LABELS)

$(ARCHIVE):
	curl -sSL -o $@ $(URL)

clean:
	rm -f $(ARCHIVE) $(MODEL) $(LABELS)

.PHONY: all clean
