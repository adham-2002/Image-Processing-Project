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

        % Display the image in a specified axes (assuming you have an axes in your GUI)
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
        errordlg(['Error converting image to grayscale: ' exception.message], 'Conversion Error', 'modal');
    end





