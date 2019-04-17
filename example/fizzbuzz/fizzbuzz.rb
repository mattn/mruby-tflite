#!mruby

def bin(n, num_digits)
  f = []
  0.upto(num_digits-1) do |x|
    f[x] = (n >> x) & 1
  end
  return f
end

def dec(b, n)
  b.each_with_index do |x, i|
    if x > 0.4
      return case i+1
      when 1; n.to_s
      when 2; 'Fizz'
      when 3; 'Buzz'
      when 4; 'FizzBuzz'
      end
    end
  end
  raise "f*ck"
end

model = TfLite::Model.from_file 'fizzbuzz_model.tflite'
interpreter = TfLite::Interpreter.new(model)
interpreter.allocate_tensors
input = interpreter.input_tensor(0)
output = interpreter.output_tensor(0)
1.upto(100) do |x|
  input.data = bin(x, 7)
  interpreter.invoke
  puts dec(output.data, x)
end
