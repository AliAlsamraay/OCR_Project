Imports Emgu.CV
Imports Emgu.CV.CvEnum
Imports Emgu.CV.Structure

Imports Numpy

Imports Keras
Imports Keras.Layers
Imports Keras.Models
Imports Keras.Optimizers

Imports Emgu.CV.Util
Imports Python.Runtime

Public Class Form1
    Public imageList As New List(Of Mat)



    Private Sub Button3_Click(sender As Object, e As EventArgs) Handles Button3.Click
        'This Function To removing the noise from image.


        Dim openFileDialog1 As New OpenFileDialog()

        ' Set the filter for image files.
        openFileDialog1.Filter = "Image Files(*.bmp;*.jpg;*.png;*.jfif)|*.bmp;*.jpg;*.png;*.jfif"
        If openFileDialog1.ShowDialog() = DialogResult.OK Then
            'Get the path of the selected file.
            Dim imagePath As String = openFileDialog1.FileName
            Dim srcImage As Mat = CvInvoke.Imread(imagePath)

            CvInvoke.CvtColor(srcImage, srcImage, ColorConversion.Bgr2Rgb)

            Dim result As New Mat(srcImage.Size, srcImage.Depth, srcImage.NumberOfChannels)
            Dim inverted_image As New Mat(srcImage.Size, srcImage.Depth, srcImage.NumberOfChannels)
            Dim gray_image As New Mat(srcImage.Size, srcImage.Depth, srcImage.NumberOfChannels)
            Dim bw_image As New Mat(srcImage.Size, srcImage.Depth, srcImage.NumberOfChannels)
            Dim desk_Image As New Mat(srcImage.Size, srcImage.Depth, srcImage.NumberOfChannels)

            'Inverted Image:
            CvInvoke.BitwiseNot(srcImage, inverted_image,)


            'Binarization:
            CvInvoke.CvtColor(srcImage, gray_image, ColorConversion.Rgb2Gray)
            CvInvoke.Threshold(gray_image, bw_image, 210, 230, ThresholdType.Binary)



            'Noise Removal
            Dim no_noise_image = noise_removal(bw_image)

            CvInvoke.Imshow("src image", srcImage)

            'Dilation and Erosion
            Dim thin_font_image = thin_font(no_noise_image)
            Dim thick_font_image = thick_font(no_noise_image)


            'Rotation - Deskewing
            desk_Image = deskew(no_noise_image)

            'Removing Borders
            Dim no_borders = removeBorders(no_noise_image)

            'Rescaling the image
            Dim rescaled_image = RescaleImage(no_borders, 1.3)







            'Define the kernel (5x5 box blur)
            Dim kernel As New Mat(5, 5, DepthType.Cv32F, 1)
            'kernel.SetTo(New MCvScalar(1.0 / (kernel.Rows * kernel.Cols)))
            'CvInvoke.Filter2D(srcImage, result, kernel, Point.Empty)




            If GaussianCheckBox.Checked Then
                'apply a Gaussian blur filter
                Dim blurSize As Integer = 5 ' Set the size of the blur kernel
                Dim blurKernel As New Size(blurSize, blurSize) ' Create a kernel with the desired size
                CvInvoke.Blur(srcImage, result, blurKernel, New Point(-1, -1), BorderType.Default)
                Dim m = New Mat
                'Apply Gaussian blur to the image to reduce noise
                CvInvoke.GaussianBlur(srcImage, m, New Size(5, 5), 0)
                CvInvoke.Imshow("gauss image", m)


                CvInvoke.WaitKey(0)
            End If




            If medianChkBox.Checked Then
                'Apply MedianBlur to the image to reduce noise
                CvInvoke.MedianBlur(result, result, 5)

                ' Add your images to the list
                imageList.Add(result)
            End If

            ' Load the image as grayscale
            Dim rgbImage As New Mat()
            CvInvoke.CvtColor(srcImage, rgbImage, ColorConversion.Bgr2Rgb)

            ' Apply bilateral 
            CvInvoke.BilateralFilter(result, rgbImage, 5, 75, 75)

            'showAllImages()
            ' Show the result
            'CvInvoke.Imshow("Final result Image", rgbImage)
            'CvInvoke.WaitKey(0)
            CvInvoke.DestroyAllWindows()
        End If

    End Sub

    Function ResizeImage(image As Mat, newSize As Size)
        Dim resizedImage = New Mat()
        CvInvoke.Resize(image, resizedImage, newSize)
        Return resizedImage
    End Function


    Function RescaleImage(image As Mat, scaleFactor As Double)
        Dim newSize As New Size(CInt(image.Width * scaleFactor), CInt(image.Height * scaleFactor))
        Return ResizeImage(image, newSize)
    End Function



    Function noise_removal(image As Mat)
        Dim dilatedImage As New Mat
        Dim erodedImage As New Mat
        Dim morphImage As New Mat
        Dim result As New Mat

        'Noise Removal:
        Dim dilKernel As New Mat(New Size(1, 1), DepthType.Cv8U, 1)
        Dim erodeKernel As New Mat(New Size(1, 1), DepthType.Cv8U, 1)

        'dilate the image
        'Point(-1,-1) represents the center of the kernel
        CvInvoke.Dilate(image, dilatedImage, dilKernel, New Point(-1, -1), 1, BorderType.Default, New MCvScalar(0))
        'erode the image
        CvInvoke.Erode(dilatedImage, erodedImage, erodeKernel, New Point(-1, -1), 1, BorderType.Default, New MCvScalar(0))

        'morphologyEx
        CvInvoke.MorphologyEx(erodedImage, morphImage, MorphOp.Close, dilKernel, New Point(-1, -1), 1, BorderType.Default, New MCvScalar(0))

        'Apply MedianBlur to the image to reduce noise
        CvInvoke.MedianBlur(morphImage, result, 3)

        Return result
    End Function


    Function thin_font(image As Mat)
        Dim inverted_image As New Mat
        Dim erodedImage As New Mat
        Dim result As New Mat


        'Inverted Image:
        CvInvoke.BitwiseNot(image, inverted_image)

        'erode the image
        Dim erodeKernel As New Mat(New Size(2, 2), DepthType.Cv8U, 1)
        CvInvoke.Erode(inverted_image, erodedImage, erodeKernel, New Point(-1, -1), 1, BorderType.Default, New MCvScalar(0))

        'Inverted Image:
        CvInvoke.BitwiseNot(erodedImage, result)

        Return result
    End Function

    Function thick_font(image As Mat)
        Dim inverted_image As New Mat
        Dim dilatedImage As New Mat
        Dim result As New Mat


        'Inverted Image:
        CvInvoke.BitwiseNot(image, inverted_image)

        'dilate the image
        Dim dilKernel As New Mat(New Size(2, 2), DepthType.Cv8U, 1)
        CvInvoke.Dilate(inverted_image, dilatedImage, dilKernel, New Point(-1, -1), 1, BorderType.Default, New MCvScalar(0))

        'Inverted Image:
        CvInvoke.BitwiseNot(dilatedImage, result)

        Return result
    End Function


    Function showAllImages()
        ' Concatenate images horizontally in pairs
        Dim row1 As New Mat()
        CvInvoke.HConcat(imageList(0), imageList(1), row1)

        Dim row2 As New Mat()
        CvInvoke.HConcat(imageList(2), imageList(3), row2)

        ' Concatenate the two rows vertically
        Dim grid As New Mat()
        CvInvoke.VConcat(row1, row2, grid)

        ' Show the concatenated image grid
        CvInvoke.Imshow("Image Grid", grid)
        CvInvoke.WaitKey(0)
    End Function


    Function GetSkewAngle(grayImage As Mat) As Single
        ' Prep image, copy, blur, and threshold
        Dim newImage As Mat = grayImage.Clone()
        Dim blur As New Mat()
        CvInvoke.GaussianBlur(newImage, blur, New Size(9, 9), 0)
        Dim thresh As New Mat()
        CvInvoke.Threshold(blur, thresh, 0, 255, ThresholdType.BinaryInv + ThresholdType.Otsu)

        ' Apply dilate to merge text into meaningful lines/paragraphs.
        ' Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
        ' But use smaller kernel on Y axis to separate between different blocks of text
        Dim kernel As Mat = CvInvoke.GetStructuringElement(ElementShape.Rectangle, New Size(30, 5), New Point(-1, -1))

        Dim dilate As New Mat()
        CvInvoke.Dilate(thresh, dilate, kernel, New Point(-1, -1), 2, BorderType.Default, New MCvScalar(0, 0, 0))


        ' Find all contours
        Dim contours As New VectorOfVectorOfPoint()
        CvInvoke.FindContours(dilate, contours, Nothing, RetrType.List, ChainApproxMethod.ChainApproxSimple)

        'Convert VectorOfVectorOfPoint to List(Of VectorOfPoint)
        Dim contoursList As List(Of VectorOfPoint) = New List(Of VectorOfPoint)()

        For i As Integer = 0 To contours.Size - 1
            Dim contour As VectorOfPoint = New VectorOfPoint()
            contour = contours.Item(i)
            contoursList.Add(contour)
        Next

        ' Perform sorting using LINQ
        contoursList = contoursList.OrderByDescending(Function(c) CvInvoke.ContourArea(c)).ToList()

        ' Loop through contours
        For Each contour As VectorOfPoint In contoursList
            Dim rect As Rectangle = CvInvoke.BoundingRectangle(contour)
            Dim x As Integer = rect.X
            Dim y As Integer = rect.Y
            Dim w As Integer = rect.Width
            Dim h As Integer = rect.Height
            CvInvoke.Rectangle(newImage, rect, New MCvScalar(0, 255, 0), 2)
        Next

        ' Draw bounding rectangles around contours
        Dim largestContour As VectorOfPoint = contoursList(0)
        CvInvoke.Rectangle(newImage, CvInvoke.BoundingRectangle(largestContour), New MCvScalar(0, 255, 0), 2)


        ' Find largest contour and surround in min area box
        Dim minAreaRect As RotatedRect = CvInvoke.MinAreaRect(largestContour)
        CvInvoke.Imwrite("temp/boxes.jpg", newImage)

        ' Determine the angle. Convert it to the value that was originally used to obtain skewed image
        Dim angle As Single = minAreaRect.Angle
        If angle < -45 Then
            angle = 90 + angle
        End If

        Return -1.0F * angle
    End Function


    Function rotateImage(ByVal cvImage As Mat, ByVal angle As Single) As Mat
        'Rotate the image around its center
        Dim newImage As Mat = cvImage.Clone()
        Dim h As Integer = newImage.Rows
        Dim w As Integer = newImage.Cols
        Dim center As New PointF(w \ 2, h \ 2)
        Dim matrix As New Matrix(Of Single)(2, 3)
        CvInvoke.GetRotationMatrix2D(center, angle, 1.0, matrix)
        CvInvoke.WarpAffine(newImage, newImage, matrix, New Size(w, h), Inter.Cubic, BorderType.Replicate)
        Return newImage
    End Function

    Function deskew(image As Mat)
        Dim angle = GetSkewAngle(image)
        If angle < -45 Then
            angle = -1 * (90 + angle)
        Else
            angle = 10 * angle
        End If

        MsgBox(angle)
        Return rotateImage(image, angle)
    End Function


    Function removeBorders(image As Mat)
        Dim contours As New VectorOfVectorOfPoint()

        CvInvoke.FindContours(image, contours, Nothing, RetrType.External, ChainApproxMethod.ChainApproxSimple)

        Dim cntsSorted As New List(Of VectorOfPoint)()
        For i As Integer = 0 To contours.Size - 1
            Dim contour As VectorOfPoint = contours(i)
            cntsSorted.Add(contour)
        Next

        cntsSorted = cntsSorted.OrderBy(Function(c) CvInvoke.ContourArea(c)).ToList()

        Dim cnt As VectorOfPoint = cntsSorted(cntsSorted.Count - 1)
        Dim rect As Rectangle = CvInvoke.BoundingRectangle(cnt)
        Dim x As Integer = rect.X
        Dim y As Integer = rect.Y
        Dim w As Integer = rect.Width
        Dim h As Integer = rect.Height

        Dim crop As New Mat(image, New Rectangle(x, y, w, h))
        Return crop
    End Function


    Private Sub Button1_Click(sender As Object, e As EventArgs) Handles Button1.Click
        'This Function to enhancing the contrast
        Dim openFileDialog1 As New OpenFileDialog()

        ' Set the filter for image files.
        openFileDialog1.Filter = "Image Files(*.bmp;*.jpg;*.png;*.jfif)|*.bmp;*.jpg;*.png;*.jfif"
        If openFileDialog1.ShowDialog() = DialogResult.OK Then
            ' Get the path of the selected file.
            Dim imagePath As String = openFileDialog1.FileName

            ' Read the image
            Dim srcImage As Mat = CvInvoke.Imread(imagePath)
            Dim dstImage As New Mat()

            ' Create a Mat object of the same size as the srcImage with all elements initialized to zero
            Dim zerosMat As New Mat(srcImage.Size, srcImage.Depth, srcImage.NumberOfChannels)
            zerosMat.SetTo(New MCvScalar(0, 0, 0))

            CvInvoke.AddWeighted(srcImage, 2.5, zerosMat, 10, 0, dstImage)

            CvInvoke.Imshow("Original vs Processed", dstImage)
            CvInvoke.WaitKey(0)

        End If
    End Sub


    ' Convert a bitmap to grayscale
    Private Function ConvertToGrayscale(bitmap As Bitmap) As Bitmap
        Dim grayscaleBitmap As New Bitmap(bitmap.Width, bitmap.Height)

        For x As Integer = 0 To bitmap.Width - 1
            For y As Integer = 0 To bitmap.Height - 1
                Dim color = bitmap.GetPixel(x, y)
                Dim grayscale = CInt(0.299 * color.R + 0.587 * color.G + 0.114 * color.B)
                grayscaleBitmap.SetPixel(x, y, Color.FromArgb(grayscale, grayscale, grayscale))
            Next
        Next

        Return grayscaleBitmap
    End Function

    ' Resize a bitmap to a specified size
    Private Function ResizeBitmap(bitmap As Bitmap, width As Integer, height As Integer) As Bitmap
        Dim resizedBitmap As New Bitmap(width, height)

        Using g = Graphics.FromImage(resizedBitmap)
            g.InterpolationMode = Drawing2D.InterpolationMode.NearestNeighbor
            g.DrawImage(bitmap, 0, 0, width, height)
        End Using

        Return resizedBitmap
    End Function


    ' Convert a bitmap to an array of pixel values
    Private Function ConvertBitmapToPixelValues(bitmap As Bitmap) As Single()
        Dim pixelValues(bitmap.Width * bitmap.Height - 1) As Single

        For x As Integer = 0 To bitmap.Width - 1
            For y As Integer = 0 To bitmap.Height - 1
                pixelValues(y * bitmap.Width + x) = bitmap.GetPixel(x, y).R / 255.0F
            Next
        Next

        Return pixelValues
    End Function

    Private Sub Button4_Click(sender As Object, e As EventArgs) Handles Button4.Click
        'this function build a neural network
        Runtime.PythonDLL = "python38.dll"
        Environment.SetEnvironmentVariable("C:\Python39\python.exe", "C:\Python39\python.exe")
        'PythonEngine.Initialize()


        Dim data = Datasets.MNIST.LoadData()

        Dim trainImages = data.Item1.Item1.reshape(60000, 28, 28, 1)
        Dim trainLabels = data.Item1.Item2.reshape(60000, 1)

        Dim testImages = data.Item2.Item1.reshape(10000, 28, 28, 1)
        Dim testLabels = data.Item2.Item2.reshape(10000, 1)

        'normalize the data
        trainImages = trainImages / 255.0F

        testImages = testImages / 255.0F





        'Create a new Sequential model, Sequential is a linear stack of layers.
        Dim model As Sequential = New Sequential()


        'Convolutional Layer 1.
        Dim filter_size1 As Integer = 3   ' Convolution filters are 3 x 3 pixels.
        Dim num_filters1 = 32  ' There are 32 Of these filters.

        'Add a Conv2D layer with 32 filters, kernel size of 3x3, ReLU activation function, and input shape of 28x28x1
        model.Add(New Conv2D(
                  num_filters1, strides:=Tuple.Create(1, 1),
                  kernel_size:=Tuple.Create(filter_size1, filter_size1),
                  activation:="relu", input_shape:=(28, 28, 1))
                  )


        'Add a MaxPooling2D layer with a pool size of 2x2
        model.Add(New MaxPooling2D(pool_size:=Tuple.Create(2, 2)))

        'Add a Dropout layer with a rate of 0.25
        'Dropout layer to prevent overfitting.
        'Overfitting occurs when a model is too complex, This can lead to poor performance.
        model.Add(New Dropout(0.25))

        'Add a Flatten layer
        'it flattens the output from the convolutional layers into a 1D vector,
        'which can then be passed to the fully connected layers for classification.
        'reshapes the output from the convolutional layers into a format that can be processed by the dense layers.
        model.Add(New Flatten())




        'Add a Dense layer with 128 neurons (Dense connect all the neurons in the current layer to all the neurons in the previous layer).
        'and ReLU activation function is simple and computationally efficient, allowing for faster training.
        'ReLU produce better results in terms of accuracy and convergence rate, It calculates max(x, 0) For Each input pixel x.
        'This adds some non-linearity To the formula And allows us To learn more complicated functions.
        model.Add(New Dense(128, activation:="relu"))

        'Add a Dropout layer with a rate of 0.5
        model.Add(New Dropout(0.5))

        'Softmax activation function is used on the last layer with 30 neurons to convert the final outputs into a probability distribution over the classes.
        'This allows us to interpret the output as the model's predicted probability for each class, and choose the class with the
        'highest probability as the final prediction.
        model.Add(New Dense(30, activation:="softmax"))

        'Adam popular and effective optimization algorithm; calculates the learning rate adaptively
        'for each parameter in the neural network, It combines the advantages of  Adagrad and RMSProp
        Dim optimizer As New Adam()

        'Sparse categorical crossentropy loss function is used when the labels are integers
        'rather than one-hot encoded vectors, which is often the case in classification problems.
        model.Compile(optimizer, loss:="sparse_categorical_crossentropy", metrics:={"accuracy"})




        ' Define the training data
        Dim x_train(,) As Double = New Double(2, 3) {
        {0.1, 0.2, 0.3, 0.4},
        {0.5, 0.6, 0.7, 0.8},
        {0.9, 0.1, 0.11, 0.12}
        }

        Dim y_train() As Double = New Double(2) {0, 1, 0}

        Dim x_test As Array = {}
        Dim y_test As Array = {}


        Dim validation_data As Array = {
            x_test,
            y_test
        }

        'train the model.
        '"epochs" refer to the number of times a dataset is passed through a neural network during training.
        model.Fit(trainImages, trainLabels, epochs:=10, validation_data:=validation_data)

        'Evaluate
        Dim score = model.Evaluate(testImages, testLabels)

        MsgBox(score)


        'get image as array



        'predict the image after convert it to array
        ' Dim y_pred = model.Predict(x_test)



    End Sub

    Private Sub Form1_Load(sender As Object, e As EventArgs) Handles MyBase.Load
        ' Initialize the Python runtime
        'PythonEngine.Initialize()

        ' Set the Python path to the location of your Python interpreter
        'PythonEngine.PythonPath = "C:\Python39\python.exe"



    End Sub
End Class

