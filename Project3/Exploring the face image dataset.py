from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

face_data = fetch_lfw_people(min_faces_per_person=80)
X = face_data.data
Y = face_data.target

print('Input data size :', X.shape)
print('Output data size :', Y.shape)
print('Label names:', face_data.target_names)

for i in range(5):
    print(f'Class {i} has {(Y == i).sum()} samples.')


fig, ax = plt.subplots(3, 4)
for i, axi in enumerate(ax.flat):
    axi.imshow(face_data.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=face_data.target_names[face_data.target[i]])

plt.show()
