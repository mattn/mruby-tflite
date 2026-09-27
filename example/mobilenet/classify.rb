#!mruby
#
# Classify an image with MobileNet V1 (quantized), the way you would use any
# image model from mruby: read the pixels, hand them to the input tensor, then
# sort the output scores.
#
#   make                                                          # get the model
#   ffmpeg -i cat.jpg -vf scale=224:224 -pix_fmt rgb24 cat.ppm    # or: magick cat.jpg -resize 224x224! cat.ppm
#   mruby classify.rb cat.ppm
#
# Any binary PPM (P6) will do; images of other sizes are scaled with nearest
# neighbour sampling.

MODEL = 'mobilenet_v1_1.0_224_quant.tflite'
LABELS = 'labels_mobilenet_quant_v1_224.txt'
TOP_N = 5
WHITESPACE = [' ', "\t", "\n", "\r"]

# PPM headers are whitespace separated tokens, with '#' starting a comment.
def next_token(f)
  token = ''
  loop do
    c = f.read(1)
    raise 'unexpected end of file' if c.nil?
    if c == '#'
      c = f.read(1) while !c.nil? && c != "\n"
    elsif WHITESPACE.include?(c)
      break unless token.empty?
    else
      token << c
    end
  end
  token
end

# Returns [width, height, RGB bytes].
def read_ppm(path)
  f = File.open(path, 'rb')
  begin
    raise "#{path}: not a binary PPM" unless next_token(f) == 'P6'
    width = next_token(f).to_i
    height = next_token(f).to_i
    max = next_token(f).to_i
    raise "#{path}: unsupported max value #{max}" unless max == 255
    [width, height, f.read(width * height * 3)]
  ensure
    f.close
  end
end

# Nearest neighbour sampling, straight into the RGB byte string the tensor wants.
def resize(pixels, src_w, src_h, dst_w, dst_h)
  out = ''
  dst_h.times do |y|
    row = (y * src_h / dst_h) * src_w
    dst_w.times do |x|
      i = (row + x * src_w / dst_w) * 3
      out << pixels[i, 3]
    end
  end
  out
end

if ARGV.empty?
  puts 'usage: classify.rb IMAGE.ppm...'
else
  labels = IO.read(LABELS).split("\n")
  interpreter = TfLite::Interpreter.new(TfLite::Model.from_file(MODEL))
  interpreter.allocate_tensors
  input = interpreter.input_tensor(0)
  output = interpreter.output_tensor(0)

  ARGV.each do |path|
    width, height, pixels = read_ppm(path)
    # Tensor#data= takes the raw bytes of a tensor as well as an array of its
    # elements; 150528 pixel components are better off as a string.
    input.data = resize(pixels, width, height, input.dim(2), input.dim(1))
    interpreter.invoke

    # The output is quantized to 0..255, so a score of 255 means 100%.
    scores = output.data
    ranking = (0...scores.size).sort { |a, b| scores[b] <=> scores[a] }
    puts path
    ranking.first(TOP_N).each do |i|
      puts "  #{(scores[i] * 100.0 / 255).round(1)}% #{labels[i]}"
    end
  end
end
