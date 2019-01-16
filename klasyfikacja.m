%% £adowanie pliku do treningu algorytmu K Nearest Neighbors

folder = 'Reuters52\';
filename = 'webkb-train.txt';
filename = strcat(folder, filename);
delimiter = '\t';

formatSpec = '%s%s%[^\n\r]';

fileID = fopen(filename,'r');

dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN, 'ReturnOnError', false);

fclose(fileID);

class_train = dataArray{:, 1};
words_train = dataArray{:, 2};

clearvars filename delimiter formatSpec fileID dataArray ans;

% £adowanie pliku do testowania algorymu KNN

filename = 'r52-test.txt';
filename = strcat(folder, filename);
delimiter = '\t';

formatSpec = '%s%s%[^\n\r]';

fileID = fopen(filename,'r');

dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN, 'ReturnOnError', false);

fclose(fileID);

class_test = dataArray{:, 1};
words_test = dataArray{:, 2};

clearvars filename delimiter formatSpec fileID dataArray ans;

%% Przygotowanie danych do treningu algorytmu KNN 
number_of_files = size(words_train);
number_of_files = number_of_files(1);

max = 0;

for i=1:number_of_files
   
     word_num = numel(strsplit(words_train{i,1}));
     if(word_num > max)
         max = word_num;
     end
end

words = cell(max, number_of_files);

%Utworzenie tablicy z wszystkimi wyrazami w tekœcie; Odrzucenie zbyt
%krótkich lub zbyt d³ugich wyrazów

for i=1:number_of_files

    raw_words = [words_train(i), repmat( {' '}, numel( words_train(i) ), 1 )]' ;
    raw_words = strtrim( [raw_words{:}] ) ;
    words_in_document = regexp((raw_words),' ','split')';
    length = size(words_in_document);
    length = length(1);

    for k=1:length
        
        word_length = size(words_in_document{k,1});
        word_length = word_length(2);
    
        if(word_length <= 2 || word_length >= 15)      
            words_in_document{k,1} = [];
        end
    end

    words_in_document = words_in_document(~cellfun('isempty',words_in_document));
    
    number_of_words = size(words_in_document);
    number_of_words = number_of_words(1);
    
    for j=1:number_of_words
        words{j,i} = words_in_document{j,1};        
    end 
end

empty_values = cellfun('isempty',words);
words(empty_values) = {'0'};   

%Utworzenie tablicy z wyrazami oraz ich czêstotliwoœci¹ wystêpowania w
%tekœcie; Odrzucenie wyrazów, które wystêpuj¹ rzadko.

[word_bank,~,id] = unique(words);
word_bank{1,1} = [];

bank_words = size(word_bank);
bank_words = bank_words(1);

word_frequency = histc(id,1:bank_words);

word_frequency_bank = [];
counter = 1;

for i=2:bank_words
    if(word_frequency(i) > 8)
       word_frequency_bank{counter,1} = word_bank{i,1};
       word_frequency_bank{counter,2} = word_frequency(i);
       counter = counter + 1;
    end    
end

clearvars counter empty_values i id j k words_in_document length 
clearvars word_frequency max word_num;

%Odrzucenie z tablicy words wyrazów nie znajduj¹cych siê w tablicy z
%rozpatrywanymi wyrazami.

words_bool_array = ismember(words,word_frequency_bank(:,1));

words_size = size(words);
rows = words_size(1);
columns = words_size(2);
new_words = words;

for i=1:columns
    for k=1:rows
        if(words_bool_array(k,i)==0)
            new_words{k,i} = [];
        end
    end
end

empty_values = cellfun('isempty',new_words);
new_words(empty_values) = {'0'};   
words = new_words;

clearvars i k rows columns empty_values new_words;

%Tworzenie tablicy z czêstotliwoœci¹ wystêpowania danego wyrazu w danym
%dokumencie (term-frequency)

tf_columns = size(word_frequency_bank);
tf_columns = tf_columns(1);
tf_rows = number_of_files;

tf = cell(tf_rows, tf_columns);

for i=1:number_of_files
    
   [w,~,id] = unique(words(:,i)); 
   temp = histc(id,1:length(w));
   
   for k=2:length(w)
      indice = find(strcmp(word_frequency_bank,w(k)));
      tf{i,indice} = temp(k);
   end
   
end

empty_values = cellfun('isempty',tf);
tf(empty_values) = {0};

%Tworzenie tablicy z odwrotn¹ iloœci¹ wyst¹pieñ danego s³owa na przestrzeni
%wszystkich dokumentów (inverse document frequency)

words_uniq = cell(1, number_of_files);

for i=1:number_of_files
    words_uniq{i} = unique(words(:,i));
end

words_temp = [];

for i=1:number_of_files
    len = numel(words_uniq{i})-1;
    counter = 2;
    for k=1:len
        words_temp{k,i} = words_uniq{1,i}{counter};
        counter = counter + 1;
    end
end

empty_values = cellfun('isempty',words_temp);
words_temp(empty_values) = {'0'};   

[word_bank,~,id] = unique(words_temp);
word_bank{1,1} = [];

bank_words = size(word_bank);
bank_words = bank_words(1);

word_frequency = histc(id,1:bank_words);

idf = cell(bank_words-1,1);

counter = 1;
for i=2:bank_words
   idf{counter,1} = log(number_of_files / word_frequency(i)); 
   counter = counter + 1;
end

%Utworzenie i obliczenie wartoœci po³¹czonych tablic tf i idf 

tfidf_train = cell(number_of_files,bank_words-1);

for i=1:number_of_files
   for k=1:bank_words-1
       tfidf_train{i,k} = tf{i,k} * idf{k,1};
   end
end

%Utworzenie i trenowanie klasyfikatora dokumentów 

cl = fitcknn(cell2mat(tfidf_train),class_train(:,1),'NumNeighbors',10);

%% Przygotowanie danych do testu wytrenowanego algorytmu

%Do testu wytrenowanych klasyfikatorów

% load r52_classifier.mat 
% load r52_word_frequency_bank
% 
% load kb_classifier.mat 
% load kb_word_frequency_bank


%Utworzenie tablicy z wszystkimi wyrazami w tekœcie; Odrzucenie zbyt
%krótkich lub zbyt d³ugich wyrazów

number_of_files = size(words_test);
number_of_files = number_of_files(1);

max = 0;

for i=1:number_of_files
   
     word_num = numel(strsplit(words_test{i,1}));
     if(word_num > max)
         max = word_num;
     end
end

words = cell(max, number_of_files);

for i=1:number_of_files

    raw_words = [words_test(i), repmat( {' '}, numel( words_test(i) ), 1 )]' ;
    raw_words = strtrim( [raw_words{:}] ) ;
    words_in_document = regexp((raw_words),' ','split')';
    length = size(words_in_document);
    length = length(1);

    for k=1:length
        
        word_length = size(words_in_document{k,1});
        word_length = word_length(2);
    
        if(word_length <= 2 || word_length >= 15)      
            words_in_document{k,1} = [];
        end
    end

    words_in_document = words_in_document(~cellfun('isempty',words_in_document));
    
    number_of_words = size(words_in_document);
    number_of_words = number_of_words(1);
    
    for j=1:number_of_words
        words{j,i} = words_in_document{j,1};        
    end 
end

empty_values = cellfun('isempty',words);
words(empty_values) = {'0'};  

%Odrzucenie z tablicy words wyrazów nie znajduj¹cych siê w tablicy z
%rozpatrywanymi wyrazami.

words_bool_array = ismember(words,word_frequency_bank(:,1));

words_size = size(words);
rows = words_size(1);
columns = words_size(2);
new_words = words;

for i=1:columns
    for k=1:rows
        if(words_bool_array(k,i)==0)
            new_words{k,i} = [];
        end
    end
end

empty_values = cellfun('isempty',new_words);
new_words(empty_values) = {'0'};
words = new_words;

%Tworzenie tablicy z czêstotliwoœci¹ wystêpowania danego wyrazu w danym
%dokumencie (term-frequency)

tf_test_columns = size(word_frequency_bank);
tf_test_columns = tf_test_columns(1);
tf_test_rows = number_of_files;

tf_test = cell(tf_test_rows, tf_test_columns);

for i=1:number_of_files
    
   [w,~,id] = unique(words(:,i)); 
   temp = histc(id,1:numel(w));
   
   for k=2:numel(w)
      indice = find(strcmp(word_frequency_bank,w(k)));
      tf_test{i,indice} = temp(k);
   end
   
end

empty_values = cellfun('isempty',tf_test);
tf_test(empty_values) = {0};

%Tworzenie tablicy z odwrotn¹ iloœci¹ wyst¹pieñ danego s³owa na przestrzeni
%wszystkich dokumentów (inverse document frequency)

words_number = size(words);
words_number = words_number(1);

words_uniq = [];

for i=1:number_of_files
    words_uniq{i} = unique(words(:,i));
end

words_temp = [];

for i=1:number_of_files
    len = numel(words_uniq{i})-1;
    counter = 2;
    for k=1:len
        words_temp{k,i} = words_uniq{1,i}{counter};
        counter = counter + 1;
    end
end

empty_values = cellfun('isempty',words_temp);
words_temp(empty_values) = {'0'};  

[word_bank,~,id] = unique(words_temp);
word_bank{1,1} = [];

bank_words = size(word_bank);
bank_words = bank_words(1);

word_frequency = histc(id,1:bank_words);

bank_words = size(word_frequency_bank);
bank_words = bank_words(1);

temp_tf = cell2mat(tf_test);

idf_test = ismember(word_frequency_bank(:,1),words);
idf_test = double(idf_test);

counter = 1;
for i=1:bank_words
   if(idf_test(i)==1)
      idf_test(i) = log(number_of_files / word_frequency(counter)); 
      counter = counter + 1;
   end
end

%Utworzenie i obliczenie wartoœci po³¹czonych tablic tf i idf 

tfidf_test = cell(number_of_files,bank_words);

for i=1:number_of_files
   for k=1:bank_words
       tfidf_test{i,k} = tf_test{i,k} * idf_test(k);
   end
end

%% Klasyfikacja testowanych dokumentów

label = predict(cl,cell2mat(tfidf_test));

%Obliczenie ile % dokumentów zosta³o sklasyfikowanych prawid³owo

good = nnz(strcmp(label,class_test));

score = (good/number_of_files)*100;
disp(score);

clearvars bank_words raw_words words_in_document colums counter empty_values;
clearvars len length new_words number_of_words rows temp temp_tf w word_length;
clearvars words_bool_array words_number words_size tf_rows tf_columns i id;
clearvars tf_test_rows tf_test_columns words_temp words_uniq indice j k;