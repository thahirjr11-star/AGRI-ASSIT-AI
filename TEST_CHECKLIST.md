# Manual verification checklist

Record each test result with date, device/browser, image source, true class, predicted class, and confidence.

- [ ] Healthy leaf: tomato
- [ ] Diseased leaf: tomato (more than one disease class)
- [ ] Healthy leaf: potato
- [ ] Diseased leaf: potato
- [ ] Healthy leaf: apple
- [ ] Diseased leaf: apple
- [ ] Healthy leaf: corn/maize
- [ ] Diseased leaf: corn/maize
- [ ] Healthy leaf: grape
- [ ] Diseased leaf: grape
- [ ] Low-confidence / non-leaf image does not produce treatment advice
- [ ] Class label and crop name match the actual model output index
- [ ] Treatment guidance corresponds to the predicted disease family
- [ ] Tamil voice input starts, asks microphone permission, and transcribes speech
- [ ] Tamil voice output reads the latest prediction
- [ ] Weather, irrigation and market demo screens open and return home
- [ ] Test on a real Android phone and on desktop browser

For an accuracy report, use labelled images not used for model training, and calculate accuracy = correct predictions / total labelled test images. Report the number and source of images; do not use training images as the test set.
