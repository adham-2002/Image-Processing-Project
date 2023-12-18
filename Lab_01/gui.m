function varargout = gui(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @gui_OpeningFcn, ...
                   'gui_OutputFcn',  @gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before gui is made visible.

function gui_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;

% Update handles structure
guidata(hObject, handles);




% --- Outputs from this function are returned to the command line.
function varargout = gui_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;


% --- Executes on button press in Import_image.
function Import_image_Callback(hObject, eventdata, handles)
    % hObject    handle to the button that was pressed
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)

    % Prompt the user to select an image file
    [fileName, filePath] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp', 'Image Files (*.png, *.jpg, *.jpeg, *.bmp)'; ...
                                      '*.*', 'All Files (*.*)'}, ...
                                      'Select an image file');

    % Check if the user selected a file or canceled the operation
    if isequal(fileName, 0)
        % User canceled the operation
        return;
    end

    % Construct the full file path
    fullFilePath = fullfile(filePath, fileName);

    try
        % Read the image
        image = imread(fullFilePath);

        % Display the image in a specified axes 
        axes(handles.original); 
        imshow(image);
        title('Imported Image');

        % Store the image data in the handles structure for future use
        handles.importedImage = image;

        % Update handles structure
        guidata(hObject, handles);
        

    catch exception
    % Display the error message
    errordlg(['Error reading the image file: ' exception.message], 'Image Import Error', 'modal');
    end
% --- Executes on button press in gray_image.
function gray_image_Callback(hObject, eventdata, handles)
    if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        % Call Rgb2Gray function to convert the imported image to grayscale
        grayImage = Rgb2Gray(handles.importedImage,2);

        % Display the grayscale image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        imshow(grayImage);
        title('Grayscale Image');

        % Store the grayscale image data in handles for future use
        handles.grayImage = grayImage;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end

% --- Executes on button press in binary_image.
function binary_image_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        % Call binary function to convert the imported image to grayscale
        binaryImage = RGB2Binary(handles.importedImage,80);

        % Display the binary image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        imshow(binaryImage);
        title('Binary Image');

        % Store the binaryimage image data in handles for future use
        handles.binaryimage = binaryImage;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end

% --- Executes on button press in negative.
function negative_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
end

    try
        % Call binary function to convert the imported image to grayscale
        negativeImage = negative_tranform(handles.importedImage);

        % Display the binary image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        imshow(negativeImage);
        title('Complement Image');

        % Store the binaryimage image data in handles for future use
        handles.negativeImage = negativeImage;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end

% --- Executes on button press in fourier.
function fourier_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        % Call binary function to convert the imported image to grayscale
        Fourierimage = FourierTransformation(handles.importedImage);

        % Display the binary image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        imshow(Fourierimage);
        title('Frequency Domain');

        % Store the binaryimage image data in handles for future use
        handles.Fourierimage = Fourierimage;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end


% --- Executes on button press in exit.
function exit_Callback(hObject, eventdata, handles)
closereq();


% --- Executes on button press in Retrieve.
function Retrieve_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
end

    try
        % Call binary function to convert the imported image to grayscale
        image = handles.importedImage;

        % Display the binary image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        imshow(image);
        title('Original Domain');

        % Store the binaryimage image data in handles for future use

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error : ' exception.message], 'Conversion Error', 'modal');
    end
% --- Executes on button press in Inverse_Log.
function Inverse_Log_Callback(hObject, eventdata, handles)


% --- Executes on button press in Log.
function Log_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        % Call binary function to convert the imported image to grayscale
        log_image = LogTransform(handles.importedImage,handles.c);

        % Display the binary image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        imshow(log_image);
        title('log image');

        % Store the binaryimage image data in handles for future use
        handles.log_image = log_image;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end


brightness_text_box_Callback
function brightness_text_box_Callback(hObject, eventdata, handles)

% Get the value from the textbox
test = get(hObject, 'String');

% Convert the string to a double
bright_dark = str2double(test);

% Check if the conversion was successful
if isnan(bright_dark)
    errordlg('Invalid value for c. Please enter a numeric value.', 'Invalid Input', 'modal');
    return;
end

% Update handles structure with the new value of c
handles.bright_dark = bright_dark;

% Update handles structure
guidata(hObject, handles);



% --- Executes during object creation, after setting all properties.
function tranformation_textbox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tranformation_textbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tranformation_textbox_Callback(hObject, eventdata, handles)
% Get the value from the textbox
c_str = get(hObject, 'String');

% Convert the string to a double
c = str2double(c_str);

% Check if the conversion was successful
if isnan(c)
    errordlg('Invalid value for c. Please enter a numeric value.', 'Invalid Input', 'modal');
    return;
end

% Update handles structure with the new value of c
handles.c = c;

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function brightness_text_box_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Add.
function Add_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
end

    try
        % Call binary function to convert the imported image to grayscale
        Bright_and_Dark = Brightness_Darkness(handles.importedImage,handles.bright_dark,1);

        % Display the binary image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        imshow(Bright_and_Dark);
        title('Bright | Dark Image');

        % Store the binaryimage image data in handles for future use
        handles.Bright_and_Dark = Bright_and_Dark;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end


% --- Executes on button press in Multiply.
function Multiply_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
end

    try
        % Call binary function to convert the imported image to grayscale
        Bright_and_Dark = Brightness_Darkness(handles.importedImage,handles.bright_dark,2);

        % Display the binary image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        imshow(Bright_and_Dark);
        title('Bright | Dark Image');

        % Store the binaryimage image data in handles for future use
        handles.Bright_and_Dark = Bright_and_Dark;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end
% --- Executes on button press in Substract.
function Substract_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
end

    try
        % Call binary function to convert the imported image to grayscale
        Bright_and_Dark = Brightness_Darkness(handles.importedImage,handles.bright_dark,3);

        % Display the binary image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        imshow(Bright_and_Dark);
        title('Bright | Dark Image');

        % Store the binaryimage image data in handles for future use
        handles.Bright_and_Dark = Bright_and_Dark;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end



% --- Executes on button press in Divide.
function Divide_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
end

    try
        % Call binary function to convert the imported image to grayscale
        Bright_and_Dark = Brightness_Darkness(handles.importedImage,handles.bright_dark,4);

        % Display the binary image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        imshow(Bright_and_Dark);
        title('Bright | Dark Image');

        % Store the binaryimage image data in handles for future use
        handles.Bright_and_Dark = Bright_and_Dark;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end

% --- Executes on button press in Powerfunction.
function Powerfunction_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
end

    try
        % Call binary function to convert the imported image to grayscale
        gamma_image = Gamma_equation(handles.importedImage,handles.c);

        % Display the binary image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        imshow(gamma_image);
        title('Gamma Image');

        % Store the binaryimage image data in handles for future use
        handles.gamma_image = gamma_image;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to Gamma Image: ' exception.message], 'Conversion Error', 'modal');
    end



function n_min_Callback(hObject, eventdata, handles)

% Get the value from the textbox
n_min = get(hObject, 'String');

% Convert the string to a double
n = str2double(n_min);

% Check if the conversion was successful
if isnan(n)
    errordlg('Invalid value for c. Please enter a numeric value.', 'Invalid Input', 'modal');
    return;
end

% Update handles structure with the new value of c
handles.n_min = n;

% Update handles structure
guidata(hObject, handles);

%function Add_Callback(hObject, eventdata, handles)
%if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
%        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
%        return;
%end


% --- Executes during object creation, after setting all properties.
function n_min_CreateFcn(hObject, eventdata, handles)
% hObject    handle to n_min (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Histogram_equalization.
function Histogram_equalization_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        % Call Rgb2Gray function to convert the imported image to grayscale
        histo_equ = histogram_equalizationGrayRGB(handles.importedImage);

        % Display the grayscale image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        imshow(histo_equ);
        title('Histogram Equalization Image');

        % Store the grayscale image data in handles for future use
        handles.histo_equ = histo_equ;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end


% --- Executes on button press in Histogram_Stretching.
function Histogram_Stretching_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
end

    try
        % Call binary function to convert the imported image to grayscale
        histo_stretch = stretching(handles.importedImage,handles.n_min,handles.n_max);
        
        % Display the binary image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        imshow(histo_stretch);
        title('Histogram Stretching');

        % Store the binaryimage image data in handles for future use
        handles.histo_stretch = histo_stretch;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end
    
function n_max_Callback(hObject, eventdata, handles)

% Get the value from the textbox
n_max = get(hObject, 'String');

% Convert the string to a double
n = str2double(n_max);

% Check if the conversion was successful
if isnan(n)
    errordlg('Invalid value for c. Please enter a numeric value.', 'Invalid Input', 'modal');
    return;
end

% Update handles structure with the new value of c
handles.n_max = n;

% Update handles structure
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function n_max_CreateFcn(hObject, eventdata, handles)
% hObject    handle to n_max (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in HE.
function HE_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        % Call Rgb2Gray function to convert the imported image to grayscale
        histo_equ = histogram_equalizationGrayRGB(handles.importedImage);
        histo_equbar = histogramFunc(histo_equ);
        % Display the grayscale image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        [H,W,L] =size(handles.importedImage);
        if L == 3
            hb = bar(histo_equbar);
            hb(1).FaceColor = 'r';
            hb(2).FaceColor = 'g';
            hb(3).FaceColor = 'b';
        else
            bar(histo_equbar);
        end
        title('Histogram Bar for Image');

        % Store the grayscale image data in handles for future use
        handles.histo_equbar = histo_equbar;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end


% --- Executes on button press in Histogram.
function Histogram_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        % Call Rgb2Gray function to convert the imported image to grayscale
        histogram = histogramFunc(handles.importedImage);

        % Display the grayscale image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        
        [H,W,L] =size(handles.importedImage);
        if L == 3
            hb = bar(histogram);
            hb(1).FaceColor = 'r';
            hb(2).FaceColor = 'g';
            hb(3).FaceColor = 'b';
        else
            bar(histogram);
        end
        title('Histogram Bar for Image');

        % Store the grayscale image data in handles for future use
        handles.histo = histogram;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end


% --- Executes on button press in Averaging.
function Averaging_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        % Call Rgb2Gray function to convert the imported image to grayscale
        Filter = LinearFilters(1);
        blurredImage = LinearFilter(handles.importedImage,Filter);

        % Display the grayscale image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        imshow(blurredImage);
        title('Averaging ( Blurred ) Image');

        % Store the grayscale image data in handles for future use
        handles.blurredImage = blurredImage;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end


% --- Executes on button press in point_detection.
function point_detection_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        Filter = LinearFilters(3);
        detectionImage = LinearFilter(handles.importedImage,Filter);

        axes(handles.axes2);
        imshow(detectionImage);
        title('Edge Detection Image');

        % Store the grayscale image data in handles for future use
        handles.detectionImage = detectionImage;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end


% --- Executes on button press in Point_Sharpening.
function Point_Sharpening_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        Filter = LinearFilters(12);
        sharpenedImage = LinearFilter(handles.importedImage,Filter);

        axes(handles.axes2);
        imshow(sharpenedImage);
        title('Sharpened Image');

        % Store the grayscale image data in handles for future use
        handles.sharpenedImage = sharpenedImage;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end



% --- Executes on button press in max_smoothing.
function max_smoothing_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        maxSmoothedImage = NonLinear(handles.importedImage,4);

        axes(handles.axes2);
        imshow(maxSmoothedImage);
        title('Max (Smoothing) Image');

        % Store the grayscale image data in handles for future use
        handles.maxSmoothedImage = maxSmoothedImage;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end

% --- Executes on button press in min_smoothing.
function min_smoothing_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        minSmoothedImage = NonLinear(handles.importedImage,1);

        axes(handles.axes2);
        imshow(minSmoothedImage);
        title('Min (Smoothing) Image');

        % Store the grayscale image data in handles for future use
        handles.minSmoothedImage = minSmoothedImage;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end


% --- Executes on button press in median_smoothing.
function median_smoothing_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        medianSmoothedImage = NonLinear(handles.importedImage,2);

        axes(handles.axes2);
        imshow(medianSmoothedImage);
        title('median (Smoothing) Image');

        % Store the grayscale image data in handles for future use
        handles.medianSmoothedImage = medianSmoothedImage;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end



% --- Executes on button press in mid_point_sharpen.
function mid_point_sharpen_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        medSharpenImage = NonLinear(handles.importedImage,3);

        axes(handles.axes2);
        imshow(medSharpenImage);
        title('Sharpened Image');

        % Store the grayscale image data in handles for future use
        handles.medSharpenImage = medSharpenImage;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end



% --- Executes on button press in sobel_horizontal.
function sobel_horizontal_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        Filter = LinearFilters(4);
        SobelHorizontal = LinearFilter(handles.importedImage,Filter);

        axes(handles.axes2);
        imshow(SobelHorizontal);
        title('Edge Detection Image');

        handles.SobelHorizontal = SobelHorizontal;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end

% --- Executes on button press in sobel_vertical.
function sobel_vertical_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        Filter = LinearFilters(5);
        SobelVertical = LinearFilter(handles.importedImage,Filter);

        axes(handles.axes2);
        imshow(SobelVertical);
        title('Edge Detection Image');

        handles.SobelVertical = SobelVertical;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end

% --- Executes on button press in sobel_RD.
function sobel_RD_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        Filter = LinearFilters(6);
        SobelRD = LinearFilter(handles.importedImage,Filter);

        axes(handles.axes2);
        imshow(SobelRD);
        title('Edge Detection Image');

        handles.SobelRD = SobelRD;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end

% --- Executes on button press in sobel_LD.
function sobel_LD_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        Filter = LinearFilters(7);
        SobelLD = LinearFilter(handles.importedImage,Filter);

        axes(handles.axes2);
        imshow(SobelLD);
        title('Edge Detection Image');

        handles.SobelLD = SobelLD;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end

% --- Executes on button press in robert_horizontal.
function robert_horizontal_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        Filter = LinearFilters(8);
        RobertHorizontal = LinearFilter(handles.importedImage,Filter);

        axes(handles.axes2);
        imshow(RobertHorizontal);
        title('Edge Detection Image');

        handles.RobertHorizontal = RobertHorizontal;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end

% --- Executes on button press in robert_vertical.
function robert_vertical_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        Filter = LinearFilters(9);
        RobertVertical = LinearFilter(handles.importedImage,Filter);

        axes(handles.axes2);
        imshow(RobertVertical);
        title('Edge Detection Image');

        handles.RobertVertical = RobertVertical;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end


% --- Executes on button press in robert_RD.
function robert_RD_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        Filter = LinearFilters(10);
        RobertRD = LinearFilter(handles.importedImage,Filter);

        axes(handles.axes2);
        imshow(RobertRD);
        title('Edge Detection Image');

        handles.RobertRD = RobertRD;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end


% --- Executes on button press in robert_LD.
function robert_LD_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        Filter = LinearFilters(11);
        RobertLD = LinearFilter(handles.importedImage,Filter);

        axes(handles.axes2);
        imshow(RobertLD);
        title('Edge Detection Image');

        handles.RobertLD = RobertLD;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end

% --- Executes on button press in point_sharpen_horizontal.
function point_sharpen_horizontal_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        Filter = LinearFilters(13);
        SharpenHorizontal = LinearFilter(handles.importedImage,Filter);

        axes(handles.axes2);
        imshow(SharpenHorizontal);
        title('Sharpened Horizontal Image');

        handles.SharpenHorizontal = SharpenHorizontal;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end

% --- Executes on button press in point_sharpen_vertical.
function point_sharpen_vertical_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        Filter = LinearFilters(14);
        SharpenVertical = LinearFilter(handles.importedImage,Filter);

        axes(handles.axes2);
        imshow(SharpenVertical);
        title('Sharpened Vertical Image');

        handles.SharpenVertical = SharpenVertical;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end

% --- Executes on button press in point_sharpen_RD.
function point_sharpen_RD_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        Filter = LinearFilters(15);
        SharpenRD = LinearFilter(handles.importedImage,Filter);

        axes(handles.axes2);
        imshow(SharpenRD);
        title('Sharpened Right Diagonal Image');

        handles.SharpenRD = SharpenRD;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end

% --- Executes on button press in point_sharpen_LD.
function point_sharpen_LD_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        Filter = LinearFilters(16);
        SharpenLD = LinearFilter(handles.importedImage,Filter);

        axes(handles.axes2);
        imshow(SharpenLD);
        title('Sharpened Left Diagonal Image');

        handles.SharpenLD = SharpenLD;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end

% --- Executes on button press in weighted_mean.
function weighted_mean_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        Filter = LinearFilters(2);
        WeightedImage = LinearFilter(handles.importedImage,Filter);

        axes(handles.axes2);
        imshow(WeightedImage);
        title('Weighted mean Image');

        % Store the grayscale image data in handles for future use
        handles.WeightedImage = WeightedImage;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to Weighted Image: ' exception.message], 'Conversion Error', 'modal');
    end


% --- Executes on button press in gray_binary.
function gray_binary_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'importedImage') || isempty(handles.importedImage)
        errordlg('Please import an image first.', 'Image Not Imported', 'modal');
        return;
    end

    try
        % Call binary function to convert the imported image to grayscale
        gray_binary = Gray2Binary(handles.importedImage,150);

        % Display the binary image in axes2 (replace 'axes2' with your actual axes tag)
        axes(handles.axes2);
        imshow(gray_binary);
        title('Binary Image');

        % Store the binaryimage image data in handles for future use
        handles.gray_binary = gray_binary;

        % Update handles structure
        guidata(hObject, handles);

    catch exception
        % Display the error message
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end