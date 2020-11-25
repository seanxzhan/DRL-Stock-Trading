from preprocess import get_data
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
#     train, test, _ = get_data()
#     timestep = 52
#     past_num = 50
#     sliced_price_history = train[:, timestep - past_num:timestep, :]
# #   closing_prices = np.reshape(sliced_price_history[:, past_num, 3],(-1,))
#     closing_prices = train[:, timestep, 3]
#
#     print(closing_prices)
#
#     inputs = tf.random.normal([1, 50, 25])
#     gru = tf.keras.layers.GRU(24)
#     output = gru(inputs)
#     print(output.shape)
#
#     gru = tf.keras.layers.GRU(24, return_sequences=True, return_state=True)
#     whole_sequence_output, final_state = gru(inputs)
#     print(whole_sequence_output.shape)
#     print(final_state.shape)

    portfolio_across_batches = np.zeros((6, 90))
    portfolio = np.array([1, 2, 3, 4, 5, 6])
    portfolio_across_batches[:, 1] = portfolio
    print(portfolio_across_batches)

