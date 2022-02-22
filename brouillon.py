


data = tf_stretch_allcond[band_prep][cond][band][n_chan, :, :]

plt.pcolormesh(time, frex, data, shading='gouraud', cmap=plt.get_cmap('seismic'))
plt.show()

np.max(data)
np.min(data)
np.mean(data)
np.std(data)

