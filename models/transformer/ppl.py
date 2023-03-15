from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='/home/mudro/Documents/Projects/feedback_gen/models/saved/v3.3')
# set_seed(7)
generated = generator("Дія.Підпис", max_length=512, num_return_sequences=5)
print(generated)
