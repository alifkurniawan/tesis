from tqdm import tqdm
from torch.utils.data import DataLoader
from rgn.model import RgnModel
from rgn.data import ProteinNet, sequence_collate
from rgn.geometric_unit import *
from datetime import datetime

# load dataset
data_path = '../data/bcolz/'
file_name = 'training_30'
trn_dataset = ProteinNet(data_path + file_name + '.bc')
val_dataset = ProteinNet(data_path+'training_30.bc')

trn_data = DataLoader(trn_dataset, batch_size=32, shuffle=True, collate_fn=sequence_collate)
val_data = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=sequence_collate)

# set model
rgn = RgnModel(42, 32, linear_units=60)
optimizer = torch.optim.Adam(rgn.parameters(), lr=1e-3)
running_loss = 0.0
drmsd = dRMSD()
train_loss = []
val_loss = []
start_time = datetime.now()
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
            train_loss.append(running_loss/i)
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
            val_loss.append(running_loss / i)
            print('Epoch {}, Val Loss {}'.format(epoch, running_loss / i))
            running_loss = 0.0

end_time = datetime.now()
with open("../output/rgn-" + file_name, 'w') as output_file:
    output_file.write(str(train_loss))
    output_file.write("\n")
    output_file.write(str(val_loss))
    output_file.write("\n")
    output_file.write(str(start_time))
    output_file.write("\n")
    output_file.write(str(end_time))
    output_file.flush()

print('Finished Training')