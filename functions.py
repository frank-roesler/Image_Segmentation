from torchvision import transforms


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20):
    """trains a model for specified number of epochs"""
    for epoch in range(epochs):
        training_loss = 0.0
        val_loss      = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input, target = batch
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * input.size(0)
        training_loss /= len(train_loader.dataset)

        model.eval()
        for batch in val_loader:
            input, target = batch
            output = model(input)
            loss = loss_fn(output, target)
            val_loss += loss.data.item() * input.size(0)
        val_loss /= len(val_loader.dataset)
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch,
                training_loss, val_loss))


def predict(model,img):
    """Given an input image and a model, this function computes and returns the model's predicted segmentation"""
    totensor = transforms.ToTensor()
    img_tensor = totensor(img.convert('RGB'))
    img_tensor = img_tensor.unsqueeze(0)
    model.eval()
    prediction = model(img_tensor)
    toimage = transforms.ToPILImage()
    prediction[prediction>0.6] = 1
    segmentation = toimage(prediction)
    return segmentation




















