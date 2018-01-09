class Config():

    def __init__(self, batch_size=28, \
                 num_epochs=25, \
                 embed_size=50, \
                 state_size=512, \
                 question_max_words=15, \
                 answer_max_words=5):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.embed_size = embed_size
        self.state_size = state_size
        self.question_max_words = question_max_words
        self.answer_max_words = answer_max_words
        self.testSize = 32768 # conv5_3 : (?, 8, 8, 512) | 32768 = 8 * 8 * 512

        self.weights_path = "../weights/vgg16_weights.npz"
        self.data_path = "../data/dataset_v7w_telling.json"
        self.glove_path = "../data/glove.6B.50d.txt"
