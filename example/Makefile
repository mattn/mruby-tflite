# Builds an mruby with this gem in it, for running the examples:
#
#   make                       # clones mruby into mruby-src, builds ./mruby
#   make MRUBY_SRC=~/mruby     # reuses an mruby checkout you already have
#   make MRUBY_VERSION=3.3.0   # a branch or tag other than master
#
# The examples then run as ../mruby <script>.rb from their own directory.

MRUBY_VERSION = master
MRUBY_SRC = mruby-src
MRUBY = mruby
CONFIG = $(abspath build_config.rb)
GEM = $(wildcard ../src/*.c) ../mrbgem.rake

all: $(MRUBY)

$(MRUBY): $(GEM) $(CONFIG) | $(MRUBY_SRC)
	cd $(MRUBY_SRC) && MRUBY_CONFIG=$(CONFIG) ./minirake
	cp $(MRUBY_SRC)/build/host/bin/mruby $@

$(MRUBY_SRC):
	git clone --depth 1 -b $(MRUBY_VERSION) https://github.com/mruby/mruby.git $@

clean:
	rm -f $(MRUBY) build_config.rb.lock

distclean: clean
	rm -rf $(MRUBY_SRC)

.PHONY: all clean distclean
