#!mruby
#
# Measure how many times a model can be invoked per second for a few thread
# counts. Shows how to load a model from a buffer, how to hand
# InterpreterOptions to an interpreter, and how to fill a tensor with raw bytes.
#
#   mruby benchmark.rb ../xor/xor_model.tflite
#
# Tiny models such as xor are dominated by call overhead, so the thread counts
# make no difference there. Point it at something like the model in
# ../mobilenet to see threads earn their keep.

BUDGET = 1.0
BATCH = 20

path = ARGV[0] || '../xor/xor_model.tflite'

# A model can be built from a buffer instead of a file, and reused by any
# number of interpreters.
model = TfLite::Model.new File.open(path, 'rb') { |f| f.read }

[1, 2, 4].each do |num_threads|
  options = TfLite::InterpreterOptions.new
  options.num_threads = num_threads

  interpreter = TfLite::Interpreter.new(model, options)
  interpreter.allocate_tensors
  # Feed zeros so that the numbers say something about the runtime, not the
  # data. Raw bytes work for any tensor type.
  interpreter.input_tensor(0).data = "\x00" * interpreter.input_tensor(0).byte_size

  count = 0
  start = Time.now
  loop do
    BATCH.times { interpreter.invoke }
    count += BATCH
    elapsed = Time.now - start
    break if elapsed >= BUDGET
  end
  elapsed = Time.now - start

  puts "#{num_threads} thread(s): #{count} invokes in #{(elapsed * 1000).round}ms " \
       "(#{(count / elapsed).round}/sec)"
end
