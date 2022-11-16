import os, random, itertools

import torch
import torch.nn as nn
from torch import optim

from const import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, MAX_LENGTH, device 

def indexes_from_sentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_TOKEN]

def zero_padding(l, fillvalue=PAD_TOKEN):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binary_matrix(l, value=PAD_TOKEN):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_TOKEN:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def input_var(l, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def output_var(l, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    mask = binary_matrix(pad_list)
    mask = torch.BoolTensor(mask)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2train_data(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    ivar, lengths = input_var(input_batch, voc)
    ovar, mask, max_target_len = output_var(output_batch, voc)
    return ivar, lengths, ovar, mask, max_target_len

def mask_nll_loss(source, target, mask):
    total = mask.sum()
    cross_entropy = -torch.log(torch.gather(source, 1, target.view(-1, 1)).squeeze(1))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, total.item()

def train(model, source, lengths, target, mask, max_target_len, clip, teacher_forcing_ratio, max_length=MAX_LENGTH):

    # Zero gradients
    model.encoder_optimizer.zero_grad()
    model.decoder_optimizer.zero_grad()

    # Set device options
    source = source.to(device)
    target = target.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lenghts = lengths.to('cpu')

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = model.encoder(source, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_TOKEN for _ in range(model.batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:model.decoder_n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequence through decoder one time step at a time
    for t in range(max_target_len):
        decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
        if use_teacher_forcing:
            # Teacher forcing: next input is current target
            decoder_input = target[t].view(1, -1)
        else:
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(model.batch_size)]])
            decoder_input = decoder_input.to(device)
        # Calculate and accumulate loss
        mask_loss, n_total = mask_nll_loss(decoder_output, target[t], mask[t])
        loss += mask_loss
        print_losses.append(mask_loss.item() * n_total)
        n_totals += n_total

    # Perform backpropagation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(model.encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(model.decoder.parameters(), clip)

    # Adjust model weights
    model.encoder_optimizer.step()
    model.decoder_optimizer.step()

    return sum(print_losses) / n_totals

def trainIters(model, pairs, save_dir, start_iteration, n_iteration, clip, teacher_forcing_ratio,
        print_every=1, save_every=500):

    # Load batches for each iteration
    training_batches = [batch2train_data(model.voc, [random.choice(pairs) for _ in range(model.batch_size)])
            for _ in range(n_iteration)]

    # Initialization
    print('\nInitializing...')
    print_loss = 0

    # Training loop
    print('\nTraining...')
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        source, lengths, target, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(model, source, lengths, target, mask, max_target_len, clip, teacher_forcing_ratio)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print('\nIteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}'.format(
                iteration, iteration/n_iteration*100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, '{}-{}_{}'.format(
                model.encoder_n_layers, model.decoder_n_layers, model.hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            model.save(os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

def run(model, pairs, save_dir, start_iteration=1, n_iteration=4000, clip=50.0, teacher_forcing_ratio=1.0):
    # Ensure dropout layers are in train mode
    model.encoder.train()
    model.decoder.train()

    # If you have cuda, configure cuda to call
    for state in model.encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in model.decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Run training iterations
    print('\nStart Training!')
    trainIters(model, pairs, save_dir, start_iteration, n_iteration, clip, teacher_forcing_ratio)
