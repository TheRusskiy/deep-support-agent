import tensorflow as tf
import numpy as np
import pickle

class Predictor:
    def init(self):
        _, vocab_to_int, int_to_vocab = self.load_preprocess()
        self.vocab_to_int = vocab_to_int
        self.int_to_vocab = int_to_vocab
        seq_length, _ = self.load_params()
        self.seq_length = seq_length

    def predict(self, initial_words, break_on_end=True, mode='most_likely'):
        gen_length = 200
        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            # Load saved model
            loader = tf.train.import_meta_graph('./save.meta')
            loader.restore(sess, './save')

            # Get Tensors from loaded model
            input_text, initial_state, final_state, keep_prob, probs = self.get_tensors(loaded_graph)

            # Sentences generation setup
            gen_sentences = self.make_initial_sentences(initial_words)
            prev_state = sess.run(initial_state, {input_text: np.array([[1]]), keep_prob: 1.0})

            # Generate sentences
            for n in range(gen_length):
                # Dynamic Input
                dyn_input = [[self.vocab_to_int[word] for word in gen_sentences[-self.seq_length:]]]
                dyn_seq_length = len(dyn_input[0])

                # Get Prediction
                probabilities, prev_state = sess.run(
                    [probs, final_state],
                    {input_text: dyn_input, initial_state: prev_state, keep_prob: 1.0})
                
                pred_word = self.pick_word(probabilities[dyn_seq_length-1], mode=mode)
                gen_sentences.append(pred_word)
                if (pred_word == '||END||') and break_on_end:
                    break
            
            # Remove tokens
            admin_reply = ' '.join(gen_sentences)
            print(admin_reply)
            return admin_reply

    def load_preprocess(self):
        """
        Load the Preprocessed Training data and return them in batches of <batch_size> or less
        """
        return pickle.load(open('./preprocess.p', mode='rb'))
        seq_length, load_dir = self.load_params()

    def get_tensors(self, loaded_graph):
        """
        Get input, initial state, final state, and probabilities tensor from <loaded_graph>
        :param loaded_graph: TensorFlow graph loaded from file
        :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
        """
        return loaded_graph.get_tensor_by_name("input:0"),\
               loaded_graph.get_tensor_by_name("initial_state:0"),\
               loaded_graph.get_tensor_by_name("final_state:0"),\
               loaded_graph.get_tensor_by_name("keep_prob:0"),\
               loaded_graph.get_tensor_by_name("probs:0")

    def pick_word(self, probabilities, mode):
        """
        Pick the next word in the generated text
        :param probabilities: Probabilites of the next word
        :param int_to_vocab: Dictionary of word ids as the keys and words as the values
        :return: String of the predicted word
        """
        if mode == 'most_likely':
            choice = np.where(probabilities==max(probabilities))[0][0]
        else:
            choice = np.random.choice(len(probabilities), 1, p=probabilities)[0]
        return self.int_to_vocab[choice]


    def load_params(self):
        """
        Load parameters from file
        """
        return pickle.load(open('params.p', mode='rb'))

    def make_initial_sentences(self, words):
        results = []
        for word in words:
            if word in self.vocab_to_int:
                results.append(word)
            else:
                results.append('||UNKNOWN||')
        return results