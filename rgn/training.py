from tqdm import tqdm
from torch.utils.data import DataLoader
from rgn.model import RgnModel
from rgn.data import ProteinNet, sequence_collate
from rgn.geometric_unit import *

# load dataset
data_path = '../data/bcolz/'
trn_dataset = ProteinNet(data_path+'training_30.bc')
val_dataset = ProteinNet(data_path+'validation.bc')

trn_data = DataLoader(trn_dataset, batch_size=32, shuffle=True, collate_fn=sequence_collate)
val_data = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=sequence_collate)

# set model
rgn = RgnModel(42, 32, linear_units=60)
optimizer = torch.optim.Adam(rgn.parameters(), lr=1e-3)
running_loss = 0.0
drmsd = dRMSD()
for epoch in range(30):
    last_batch = len(trn_data) - 1
    for i, data in tqdm(enumerate(trn_data)):
        names = data['name']
        coords = data['coords']
        mask = data['mask']

        optimizer.zero_grad()
        outputs = rgn(data['sequence'], data['length'])

        loss = drmsd(outputs, coords, mask)

        loss.backward()
        nn.utils.clip_grad_norm_(rgn.parameters(), max_norm=50)
        optimizer.step()

        running_loss += loss.item()
        if (i != 0) and (i % last_batch == 0):
            print('Epoch {}, Train Loss {}'.format(epoch, running_loss / i))
            running_loss = 0.0
            break

    last_batch = len(val_data) - 1
    for i, data in tqdm(enumerate(val_data)):
        names = data['name']
        coords = data['coords']
        mask = data['mask']

        outputs = rgn(data['sequence'], data['length'])
        loss = drmsd(outputs, coords, mask)

        running_loss += loss.item()
        if (i != 0) and (i % last_batch == 0):
            print('Epoch {}, Val Loss {}'.format(epoch, running_loss / i))
            running_loss = 0.0

print('Finished Training')