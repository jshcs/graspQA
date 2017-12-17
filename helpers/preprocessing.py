import json


def pre_process(file_name):
    """
    Tokenize all questions, answers, and multiple choices. All punctuations removed. 
	
    Pre processing the dataset. Load the dataset. 
    1. Reads the json file. 
    2. Loads all the data into a list. 
    3. This method zips all the preprocessing material and return it. 
    """

    data = json.load(open(file_name))
    data = data['images']

    images = []
    questions = []
    answers = []
    options1 = []
    options2 = []
    options3 = []
    options4 = []

    for i in range(len(data)):

        qa_pairs, image_id, split, file_name = data[i].keys()

        d = data[i]

        dqa = d['qa_pairs']
        image = d['filename']

        for j in range(len(dqa)):
            images.append(image)
            questions.append(dqa[j]['question'])
            answers.append(dqa[j]['answer'])
            options1.append(dqa[j]['multiple_choices'][0])
            options2.append(dqa[j]['multiple_choices'][1])
            options3.append(dqa[j]['multiple_choices'][2])
            options4.append(dqa[j]['answer'])

    return list(zip(images, questions, answers, options1, options2, options3, options4))


def loadData(file_name):
    """
    loadData loads the data, 
    1. Image
    2. Question
    3. Answer
    4. Option1 - Multiple choice
    5. Option2 - Multiple choice
    6. Option3 - Multiple choice
    """
    samples = pre_process(file_name)
    return samples
