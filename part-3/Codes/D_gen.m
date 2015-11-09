function [D] = D_gen(folder)

folder = strcat(folder,'/');
files = dir(folder);
count = 0;
for k = 3:size(files,1)
    fname = files(k).name;
    if strcmp( fname(size(fname,2)-3:size(fname,2)) ,'.jpg'  ) ~= 1
        continue;
    end
    count = count +1;
    D(count,count) = 0;
    I = imread(strcat(folder, fname));
    cindex = count;
    for i = k+1:size(files,1)
        cname = files(i).name;
        if strcmp( cname(size(fname,2)-3:size(fname,2)) ,'.jpg'  ) ~= 1
            continue;
        end
        cindex = cindex+1;
        cI = imread(strcat(folder, cname));
        tempI = I-cI;
        dist = sqrt(sum((sum(sum(tempI.*tempI)))));
        D(count,cindex) = dist;
        D(cindex,count) = dist;
        if (cindex/20 == floor(cindex/20))
           cindex; 
        end
    end
    count;
end
feature memstats