
# Plot features using predictions to color datapoints
plt.scatter(X[:, 0], X[:, 1], c=Z, edgecolors='k', cmap=plt.cm.Paired); plt.show()

# Run prediction on the img_idx-th image
img_idx = 12

Xi = hp.extract_img_features(image_dir + files[img_idx],patch_size)
Zi = logreg.predict(Xi)
plt.scatter(Xi[:, 0], Xi[:, 1], c=Zi, edgecolors='k', cmap=plt.cm.Paired)

# Display prediction as an image
w = gt_imgs[img_idx].shape[0]
h = gt_imgs[img_idx].shape[1]
predicted_im = hp.label_to_img(w, h, patch_size, patch_size, Zi)
cimg = hp.concatenate_images(imgs[img_idx], predicted_im)
fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size 
plt.imshow(cimg, cmap='Greys_r');
plt.title('Prediction')
plt.show()

new_img = hp.make_img_overlay(imgs[img_idx], predicted_im)

plt.imshow(new_img);
plt.title('Prediction')
plt.show()

#Make submission
masks_to_submission('submission_test.csv', *image_filenames)
