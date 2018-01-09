import json


def pre_process(file_name):
    """
    Tokenize all questions, answers, and multiple choices. All punctuations removed. 
	
    Pre processing the dataset. Load the dataset. 
    1. Reads the json file. 
    2. Loads all the image_objects into a list.
    3. This method zips all the preprocessing material and return it. 
    """

    # @TODO: Also store the Question type | Useful information to create Question-wise Stats

    dataset = json.load(open(file_name))
    print("Finished loading ", dataset["dataset"], " dataset")
    image_objects = dataset['images']

    images = []
    questions = []
    answers = []
    options1 = []
    options2 = []
    options3 = []
    options4 = []

    for image_object in image_objects:

        image_filename = image_object['filename']
        qa_pairs = image_object['qa_pairs']

        for qa_pair in qa_pairs:
            images.append(image_filename)
            questions.append(qa_pair['question'])
            answers.append(qa_pair['answer'])
            options1.append(qa_pair['multiple_choices'][0])
            options2.append(qa_pair['multiple_choices'][1])
            options3.append(qa_pair['multiple_choices'][2])
            options4.append(qa_pair['answer'])

    return list(zip(images, questions, answers, options1, options2, options3, options4))


def loadData(file_name):
    """
    loadData loads the image_objects,
    1. Image
    2. Question
    3. Answer
    4. Option1 - Multiple choice
    5. Option2 - Multiple choice
    6. Option3 - Multiple choice
    """
    samples = pre_process(file_name)
    return samples
