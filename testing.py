from DataGenerator import data_generator, batch_events_merge, Event_batching
from PointProcess import MultiVariateHawkesProcessModel
import numpy as np
import torch
from torch.nn import functional as F

#Data is generated using tick package
decays = [.3]
baseline = [0.5, 0.1, 0.5]
adjacency = [ [[0.1], [.0], [0]],
              [[0],  [.3] , [0]],
              [[0],  [0],  [0.5]] ]

end_time = 100
n_realizations = 100


N_events = data_generator(baseline, decays, \
                          adjacency,end_time, n_realizations )

#Batching Events with appropriate masks
event_merge = batch_events_merge(N_events)
batch_events, batch_event_type, batch_pad = Event_batching(event_merge)



baseline = torch.FloatTensor(baseline).cuda()
adjacency = torch.FloatTensor(adjacency).cuda()
decays = torch.FloatTensor(decays).cuda()


event_times = torch.FloatTensor(batch_events).cuda()
event_types = torch.LongTensor(batch_event_type).cuda()
input_mask  = torch.IntTensor(batch_pad).cuda()
t0, t1 =  torch.tensor([0]).cuda(), torch.tensor([end_time]).cuda()



num_type = baseline.shape[0]
num_decay = decays.shape[0]
print("No of types=", num_type)
print("Nof of decay=", num_decay)
model = MultiVariateHawkesProcessModel(num_type, num_decay).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

loss_prev = 0
for epoch in range(5000):

    optimizer.zero_grad()
    # loglik = model(events_batched, input_masks, torch.tensor([0]).cuda(), torch.tensor([end_time]).cuda())
    (lbda, compensator) = model(event_times, \
                                event_types, input_mask, t0, t1)
    loglik = lbda - compensator
    loss = loglik.mean() * (-1)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        estimation_error = torch.norm(F.softplus(model.mu).detach() - baseline) + \
                           torch.norm(F.softplus(model.alpha).detach() - adjacency) + \
                           torch.norm(F.softplus(model.beta).detach() - decays)
        loss_curr = loss.item()
        print("At Epoch {0: } Loss= {1:.4f} Error Parameter Estitmation={2:.4f}".format(epoch, loss_curr, estimation_error.item()) )
        if abs(loss_prev - loss_curr) < 0.001:
            break
        loss_prev = loss_curr

print("Original Baseline Parameter: ", baseline)
print("Estimated Baseline Parameter: ", F.softplus(model.mu).detach())

print("Original Decay Parameter: ", decays)
print("Estimated Decay Parameter: ", F.softplus(model.beta).detach())

print("Original Adjacency Parameter: ", adjacency)
print("Estimated Adjacency Parameter: ", F.softplus(model.alpha).detach() )






