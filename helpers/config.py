class Config():

    def __init__(self, batch_size=64, \
                 num_epochs=25, \
                 embed_size=50, \
                 state_size=512, \
                 question_maxLength=15, \
                 answer_maxLength=5):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.embed_size = embed_size
        self.state_size = state_size
        self.question_maxLength = question_maxLength
        self.answer_maxLength = answer_maxLength
        self.testSize = 32768

        self.weights_path = "../weights/vgg16_weights.npz"
        self.data_path = "../data/dataset_v7w_telling.json"
        self.glove_path = "../data/glove.6B.50d.txt"
