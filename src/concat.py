import torch

#%%
tweets = torch.load('data/tweets_tensor.pt')
des = torch.load('data/des_tensor.pt')
print("tweets shape {}".format(tweets.shape))
print("des shape {}".format(des.shape))

# des = torch.load('data/des_tensor.pt')[:11826]
# print("des shape {}".format(des.shape))

#%%
# des = des.unsqueeze(1)
text = torch.cat((tweets, des), dim=1)
#%%
torch.save(text, 'data/text.pt')
