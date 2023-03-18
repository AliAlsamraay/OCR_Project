Imports System.Collections.Immutable
Imports System.Drawing.Imaging
Imports System.Net.Mime.MediaTypeNames
Imports System.Runtime.InteropServices
Imports Emgu.CV
Imports Emgu.CV.CvEnum
Imports Emgu.CV.Structure
Imports Emgu.CV.ML

Imports Microsoft.ML
Imports Microsoft.ML.Data


Imports Numpy

Imports Emgu.TF




Public Class Form1

    Private Function convertAndShowToGray(imagePath As String)
        ' Convert the image to grayscale.
        Dim grayMatrix As New ColorMatrix({
            New Single() {0.299F, 0.299F, 0.299F, 0, 0},
            New Single() {0.587F, 0.587F, 0.587F, 0, 0},
            New Single() {0.114F, 0.114F, 0.114F, 0, 0},
            New Single() {0, 0, 0, 1, 0},
            New Single() {0, 0, 0, 0, 1}
        })




        Dim grayAttributes As New ImageAttributes()
        grayAttributes.SetColorMatrix(grayMatrix)
        Dim grayImage As New Bitmap(imagePath)
        Using g As Graphics = Graphics.FromImage(grayImage)
            g.DrawImage(grayImage, New Rectangle(0, 0, grayImage.Width, grayImage.Height), 0, 0, grayImage.Width, grayImage.Height, GraphicsUnit.Pixel, grayAttributes)
        End Using

        ' Display the grayscale image in the picture box.
        pbImage.Image = grayImage
    End Function


    Public imageList As New List(Of Mat)


    Private Sub Button3_Click(sender As Object, e As EventArgs) Handles Button3.Click
        'This Function To removing the noise from image.

        Dim openFileDialog1 As New OpenFileDialog()

        ' Set the filter for image files.
        openFileDialog1.Filter = "Image Files(*.bmp;*.jpg;*.png;*.jfif)|*.bmp;*.jpg;*.png;*.jfif"
        If openFileDialog1.ShowDialog() = DialogResult.OK Then
            ' Get the path of the selected file.
            Dim imagePath As String = openFileDialog1.FileName
            Dim matImage As Mat = CvInvoke.Imread(imagePath)

            CvInvoke.CvtColor(matImage, matImage, ColorConversion.Bgr2Rgb)

            ' Define the kernel (5x5 box blur)
            Dim kernel As New Mat(5, 5, DepthType.Cv32F, 1)
            kernel.SetTo(New MCvScalar(1.0 / (kernel.Rows * kernel.Cols)))


            Dim result As New Mat(matImage.Size, matImage.Depth, matImage.NumberOfChannels)
            CvInvoke.Filter2D(matImage, result, kernel, Point.Empty)

            ' Add your images to the list
            imageList.Add(result)

            'apply a Gaussian blur filter
            Dim blurSize As Integer = 5 ' Set the size of the blur kernel
            Dim blurKernel As New Size(blurSize, blurSize) ' Create a kernel with the desired size
            CvInvoke.Blur(result, result, blurKernel, New Point(-1, -1), BorderType.Default)


            ' Add your images to the list
            imageList.Add(result)


            'Apply Gaussian blur to the image to reduce noise
            CvInvoke.GaussianBlur(result, result, New Size(5, 5), 0)

            ' Add your images to the list
            imageList.Add(result)


            'Apply MedianBlur to the image to reduce noise
            CvInvoke.MedianBlur(result, result, 5)

            ' Add your images to the list
            imageList.Add(result)


            ' Load the image as grayscale
            Dim rgbImage As New Mat()
            CvInvoke.CvtColor(matImage, rgbImage, ColorConversion.Bgr2Rgb)


            ' Apply bilateral 
            CvInvoke.BilateralFilter(result, rgbImage, 5, 75, 75)



            'showAllImages()
            ' Show the result
            CvInvoke.Imshow("Final result Image", rgbImage)
            CvInvoke.WaitKey(0)
            CvInvoke.DestroyAllWindows()
        End If
    End Sub


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
        'this function build a nural network
        Dim model As New Keras.Models.Sequential()
        model.Add(New Keras.Layers.Conv2D(512, kernel_size:=Tuple.Create(3, 3), activation:="relu", input_shape:=(100, 100, 1)))
        model.Add(New Keras.Layers.MaxPooling2D(pool_size:=Tuple.Create(2, 2)))
        model.Add(New Keras.Layers.Dense(32, input_shape:=(10, 10), activation:="relu"))
        model.Add(New Keras.Layers.Dense(1, activation:="sigmoid"))
        model.Compile(optimizer:="rmsprop", loss:="binary_crossentropy", metrics:={"accuracy"})

    End Sub
End Class

