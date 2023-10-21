$(document).ready(function () {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const photo = document.getElementById('photo');
    const name = document.getElementById('name');
    const resultImage = document.getElementById('result'); // Add this line
    const snapButton = document.getElementById('snap');
    const uploadButton = document.getElementById('upload');

    const constraints = {
        video: true
    };

    // Access the user's camera
    navigator.mediaDevices.getUserMedia(constraints)
        .then((stream) => {
            video.srcObject = stream;
        })
        .catch((error) => {
            console.error('Error accessing the camera:', error);
        });

    snapButton.addEventListener('click', function () {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        const image = canvas.toDataURL('image/png');
        photo.src = image;
        uploadButton.style.display = 'block';
    });

    uploadButton.addEventListener('click', function () {
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: { image: photo.src },
            success: function (response) {
                console.log('Image uploaded:', response.image);
                alert('Image uploaded successfully!');
                // Update the image and name in your HTML
                resultImage.src = response.image;
                name.textContent = 'Name: ' + response.name;
                resultImage.style.display = 'block'; // Show the result image
            },
            error: function (error) {
                console.error('Error uploading image:', error);
                alert('Failed to upload the image.');
            }
        });
    });
});
