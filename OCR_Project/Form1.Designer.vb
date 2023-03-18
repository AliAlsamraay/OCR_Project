<Global.Microsoft.VisualBasic.CompilerServices.DesignerGenerated()>
Partial Class Form1
    Inherits System.Windows.Forms.Form

    'Form overrides dispose to clean up the component list.
    <System.Diagnostics.DebuggerNonUserCode()>
    Protected Overrides Sub Dispose(ByVal disposing As Boolean)
        Try
            If disposing AndAlso components IsNot Nothing Then
                components.Dispose()
            End If
        Finally
            MyBase.Dispose(disposing)
        End Try
    End Sub

    'Required by the Windows Form Designer
    Private components As System.ComponentModel.IContainer

    'NOTE: The following procedure is required by the Windows Form Designer
    'It can be modified using the Windows Form Designer.  
    'Do not modify it using the code editor.
    <System.Diagnostics.DebuggerStepThrough()>
    Private Sub InitializeComponent()
        Dim resources As ComponentModel.ComponentResourceManager = New ComponentModel.ComponentResourceManager(GetType(Form1))
        pbImage = New PictureBox()
        Button3 = New Button()
        Button1 = New Button()
        Button2 = New Button()
        Button4 = New Button()
        CType(pbImage, ComponentModel.ISupportInitialize).BeginInit()
        SuspendLayout()
        ' 
        ' pbImage
        ' 
        pbImage.BackColor = Color.FromArgb(CByte(64), CByte(64), CByte(64))
        pbImage.InitialImage = CType(resources.GetObject("pbImage.InitialImage"), Image)
        pbImage.Location = New Point(374, 95)
        pbImage.Name = "pbImage"
        pbImage.Size = New Size(727, 596)
        pbImage.SizeMode = PictureBoxSizeMode.Zoom
        pbImage.TabIndex = 1
        pbImage.TabStop = False
        ' 
        ' Button3
        ' 
        Button3.Font = New Font("Segoe UI", 12F, FontStyle.Regular, GraphicsUnit.Point)
        Button3.Location = New Point(61, 776)
        Button3.Name = "Button3"
        Button3.Size = New Size(249, 133)
        Button3.TabIndex = 5
        Button3.Text = "Open an Image To Remove Noise"
        Button3.UseVisualStyleBackColor = True
        ' 
        ' Button1
        ' 
        Button1.Font = New Font("Segoe UI", 15F, FontStyle.Regular, GraphicsUnit.Point)
        Button1.Location = New Point(374, 776)
        Button1.Name = "Button1"
        Button1.Size = New Size(239, 133)
        Button1.TabIndex = 6
        Button1.Text = "enhancing the contrast"
        Button1.UseVisualStyleBackColor = True
        ' 
        ' Button2
        ' 
        Button2.Font = New Font("Segoe UI", 14F, FontStyle.Regular, GraphicsUnit.Point)
        Button2.Location = New Point(657, 777)
        Button2.Name = "Button2"
        Button2.Size = New Size(217, 133)
        Button2.TabIndex = 7
        Button2.Text = "Find Noise Type"
        Button2.UseVisualStyleBackColor = True
        ' 
        ' Button4
        ' 
        Button4.Font = New Font("Segoe UI", 14.1428576F, FontStyle.Bold, GraphicsUnit.Point)
        Button4.Location = New Point(968, 776)
        Button4.Name = "Button4"
        Button4.Size = New Size(262, 134)
        Button4.TabIndex = 8
        Button4.Text = "Build CCN"
        Button4.UseVisualStyleBackColor = True
        ' 
        ' Form1
        ' 
        AutoScaleDimensions = New SizeF(12F, 30F)
        AutoScaleMode = AutoScaleMode.Font
        BackColor = Color.DimGray
        ClientSize = New Size(1469, 1025)
        Controls.Add(Button4)
        Controls.Add(Button2)
        Controls.Add(Button1)
        Controls.Add(Button3)
        Controls.Add(pbImage)
        Name = "Form1"
        Text = "Form1"
        CType(pbImage, ComponentModel.ISupportInitialize).EndInit()
        ResumeLayout(False)
    End Sub
    Friend WithEvents pbImage As PictureBox
    Friend WithEvents Button3 As Button
    Friend WithEvents Button1 As Button
    Friend WithEvents Button2 As Button
    Friend WithEvents Button4 As Button
End Class
