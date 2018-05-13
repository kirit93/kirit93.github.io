---
layout: post
title: LSTM Neural Networks
category: blog
---

Apologies for ending the previous post a tad bit abruptly, seemed to have ran out of juice there. Now I’m back to continue where I left off from with a simple way to understand LSTM Neural Networks. Since LSTMs are all the rage now, this is a good way to get started working on them.

LSTMs are sort of RNNs on steroids. The biggest advantage it gives you over an RNN is its ability to learn longer sequences of data. For example if you were training a network to learn english sentences, an RNN wouldn’t really be able to learn that something way back at the start of a sentence is very much related to something that could be at the very end, LSTMs don’t have that issue. Well, I won’t say don’t have that issue at all, but it does much better than your vanilla RNN. The fancy Machine Learning term you need to keep in mind is ‘Vanishing Gradient’. That’s why RNNs can’t learn long sequences and combating the vanishing gradient is how the LSTM can learn long sequences. 

Now to understand how the LSTM manages this, think of the entire network as a gated circuit.

![_config.yml]({{ site.baseurl }}/images/LSTM_gates.png)

This is one step of an LSTM. This diagram shows the flow from when you have in input x at time t all the way to getting an output o at time t. All the stuff in between is what helps the LSTM learn the incoming sequences well.

I know, that’s a lot of variables, but don’t worry, it’s really easy once you understand what they really mean.

Think of it this way, when an LSTM sees a long sequence, it divides it into pieces with each piece corresponding to a time step. 
Eg. “This is so cool.” Would correspond to 16 time steps, each time step corresponding to one character which will be x at time t in the diagram. At each time step, that entire circuit in the diagram will happen.

So when an LSTM gets an input x, it needs to decide a couple of things. How much of that input should it consider while learning, how much of what came before the current letter should it consider and how much of what it already knows should it forget. The LSTM has a memory cell which is a combination of the previous memory after forgetting some stuff and learning some things from the new input. The hidden state h is important because that is the link between each of the time steps. Using the hidden state of the previous step, the current input, output etc are calculated. That’s why you see h at time t - 1 going into all those gates in the diagram. 

The fundamental idea is that at each step, the network learns how much it should consider the input and how much it should consider what happened before it. Thanks to the gating mechanism with the forget and memory cell, it can learn longer sequences than an RNN. A vanilla RNN doesn’t have this complexity so it’s easy for information early on in a sequence to get lost amongst all the computation. But since there are dedicated ‘neurons’ to handle these things in the LSTM, that problem isn't faced. 

LSTMs are used for a ton of things. Speech recognition, image recognition etc. Google and Baidu amongst other big tech companies are extensively using LSTMs in their products. 


