%% ----------------------------------------------------------------------------------
%% Implementing shape-adaptive tensor factorization (SATF) for University of Pavia data sets
%% ----------------------------------------------------------------------------------

close all
clear all
clc

%% using the website data set
%% real data set
load PaviaU;
img = paviaU;
clear paviaU;

sz = size(img);
img_src = double(img);
clear img;

img_src = reshape(img_src,sz(1)*sz(2),sz(3));
Ydata_all_DR = normcols(img_src');

Ydata_all_DR = Ydata_all_DR';
%% ground truth image
load PaviaU_gt;
img_gt = reshape(paviaU_gt,sz(1)*sz(2),1);
img_gt = double(img_gt);
trainall = [find(img_gt~=0) img_gt(find(img_gt~=0))];
trainall = trainall';
clear paviaU_gt;
clear img_gt;

classifier_type_all=cellstr(strvcat('LORSAL'));

%% parameters
MMiter  = 10;  % number of independent runs
per_class=[0.05];% ratio for train
no_classes=9;
Nofeat = 100;
neighorhood_method = 'SA';%fix | SA 
patchSize_all = [13];
DR_method_all = cellstr(strvcat('MLSVD'));
show_CMap = 'no'; % yes | no
ranking = 'yes';

%% Generate neighborhood
for i_patchSize = 1:length(patchSize_all)
    patchSize = patchSize_all(i_patchSize);
    if strcmp(neighorhood_method,'fix')
        %% 3D tensor representation for training samples
        Ydata_all_DR = reshape(Ydata_all_DR,sz(1),sz(2),sz(3));
        
        padcam = padarray(Ydata_all_DR,[patchSize patchSize],'symmetric','both');
        %figure;imagesc(padcam(:,:,10));axis image;
        
        [m,n] = size(padcam);
        ImagePatches = zeros(patchSize,patchSize,sz(3),sz(1)*sz(2));
        
        count = 1;
        for j = patchSize+1:sz(2)+patchSize
            for i = patchSize+1:sz(1)+patchSize
                temp = padcam(i-(patchSize-1)/2:i+(patchSize-1)/2, j-(patchSize-1)/2:j+(patchSize-1)/2,:);
                ImagePatches(:,:,:,count) = temp;
                %figure(2);imagesc(ImagePatches(:,:,100,count));axis image;
                count = count +1;
                
            end
        end
    elseif strcmp(neighorhood_method,'SA')
        %% 引入空间信息 for test data / 全部原始数据
        img_norm_OLD=Ydata_all_DR';
        
        estimate_sigma=1;     %% estimate noise std from image?            (in this demo leave =0)
        do_wiener=1;          %% enables Wiener-filter stage                          (recommended =1)
        
        speedup_factor=1;   %% any number between 1 and 5. speedup_factor>1 enables some speed-ups in the algorithm
        figures_y_hats=1;  %% enables display of noisy image and SA-DCT estimates      (recommended =1)
        compute_errors=1;  %% enables calculation of error criteria (PSNR, MSE, etc.)  (recommended =1)
        
        blocksize=0;   %% Constrain block-size  (recommended =0)
        coef_align=1;  %% enable or disable coefficient alignment (recommended =1)
        
        h1full_all =  [1 2 3 4 5 6 7 8 9 10 11 12 13];   %% complete set of scales to be used for hard-thresholding      (recommended =[1 2 3 5 7 9 12])
        h1Wfull_all = [1 2 3 4 5 6 7 8 9 10 11 12 13];      %% complete set of scales to be used for Wiener-filter          (recommended =[1 2 3 5 6 8])
        sharparams_all = [-0.75 -0.70 -0.85 -0.90 -0.97 -1 -1 -1 -1 -1 -1 -1 -1];  %% order-mixture parameters (define the balance between zero- and first- order polynomial fitting for each scale)  (recommended =[-0.75 -0.70 -0.85 -0.9 -0.97 -1 -1])
        
        temp = round(patchSize/2);
        h1full = h1full_all(1:temp);
        h1Wfull = h1Wfull_all(1:temp);
        sharparams = sharparams_all(1:temp);
        
        DCTthrCOEF=0.77;    %% threshold-parameter  (recommended =0.77)
        max_overlap=70;         %% limits overcompleteness (speed-up)  (recommended =70)  (note: lowering the value leads to a considerable speed-up in Pointwise SA-DCT hard-thresholding)
        max_overlapW=70;        %% limits overcompleteness (speed-up)  (recommended =70)  (note: lowering the value leads to a considerable speed-up in Pointwise SA-DCT Wiener-filtering)
        % speedup_factor=1 下面的小循环不执行
        if speedup_factor~=1   %%%  Speed-ups for "filtering performance vs. complexity" trade-off 加速“过滤性能与复杂性”之间的权衡
            h1full=h1full(round(1:5/round((6-speedup_factor)):6));              %% complete set of scales to be used for hard-thresholding
            h1Wfull=h1Wfull(round(1:4/round((5-(4/5)*speedup_factor)):5));      %% complete set of scales to be used for Wiener-filter
            max_overlap=round((6-speedup_factor).^2.64);         %% limits overcompleteness (speed-up)  (note: lowering the value leads to a considerable speed-up in Pointwise SA-DCT hard-thresholding)
            max_overlapW=ceil(0.25*(6-speedup_factor).^3.5);     %% limits overcompleteness (speed-up)  (note: lowering the value leads to a considerable speed-up in Pointwise SA-DCT Wiener-filtering)
        end
        
        %% pca 为了显示image，但不局限于
        Sigma_temp = cov(img_src);   %=n*band      ens_Xtrain'      为什么是对ens_Xtrain'提取主成分呢
        [eigvector,eigvalue] = eig(Sigma_temp);
        [eigvalue,ind] = sort(diag(eigvalue),'descend');
        eigvector = eigvector(:,ind);
        %             [eigvector,eigvalue] = PCA(ens_Xtrain',options);
        z = img_norm_OLD'*eigvector(:,1);
        z=im2double(z);
        z=reshape(z,sz(1),sz(2));
        
        %%
        if estimate_sigma
            sigma=function_stdEst2D(z,2);    %%% estimates standard deviation from image (assumes perfect AWGN model) 估计图像的标准偏差（假设完美的AWGN模型）
        else
            sigma=sigma_noise;
        end
        gammaICI=max(0.8,2.4./(log(1+60*sigma)));    %%% the maximum is used in order to ensure large enough gamma for strong levels of noise (above sigma=75) 最大值用于确保足够大的伽马值以获得强噪声（高于sigma = 75）
        %---------------------------------------------------------
        % Definition of the set of scales h1
        %---------------------------------------------------------
        if (sigma>(40/255))||(speedup_factor~=1)     %%% if noise std is large use one extra scale (slows down the algorithm) 如果噪声std很大，则使用一个额外的比例（减慢算法速度）
            h1=h1full;            %%% set of scales to be used in the hard-thresholding algorithm
            h1W=h1Wfull;          %%% set of scales to be used in the Wiener-filter algorithm
        else                        %%% if noise std is not too large, fewer scales can be used (speed-up)  %%%如果噪音标准不是太大，可以使用更少的标度（加速）
            h1=h1full(find(h1full<15));    %%% set of scales to be used in the hard-thresholding algorithm  %%%在硬阈值算法中使用的标度集 10
            h1W=h1Wfull(find(h1Wfull<7));   %%% set of scales to be used in the Wiener-filter algorithm     %%%在Wiener-filter算法中使用的标度集
        end
        lenh=numel(h1);
        %---------------------------------------------------------
        % Kernels construction
        %---------------------------------------------------------
        clear gh
        % calling kernel creation function
        [kernels, kernels_higher_order]=function_CreateLPAKernels([0 0],h1,ones(size(h1)),10,1,1,ones(2,lenh),1);
        [kernelsb, kernels_higher_orderb]=function_CreateLPAKernels([1 0],h1,ones(size(h1)),10,1,1,ones(2,lenh),1);
        for s2=1:lenh     % kernel size index
            gha=kernels_higher_order{1,s2,1}(:,:,1);   % gets single kernel from the cell array (ZERO ORDER)
            ghb=kernels_higher_orderb{1,s2,1}(:,:,1);  % gets single kernel from the cell array (FIRST ORDER)
            gh{s2}=(1+sharparams(s2))*ghb-sharparams(s2)*gha; % combines kernels into "order-mixture" kernel
            gh{s2}=single(gh{s2}((end+1)/2,(end+1)/2:end));
        end
        %---------------------------------------------------------
        % Anisotropic LPA-ICI
        %---------------------------------------------------------
        h_opt_Q=function_AnisLPAICI8(single(z),gh,single(sigma),single(gammaICI));   %%% Anisotropic LPA-ICI scales for 8 directions
        %---------------------------------------------------------
        % SA for CLSUnSAL
        %---------------------------------------------------------
        img_norm_OLD=reshape(img_norm_OLD',sz(1),sz(2),sz(3));
        
        z=single(z);
        sz=size(img_norm_OLD);
        [size_z_1,size_z_2,size_z_3]=size(z);
        if size_z_3>1
            disp(' input image must be grayscale !!! ')
            return
        end
        z=z-max(min(z(:)),0);  %% normalization
        z=z/min(1,max(z(:)));  %% normalization
        FrameZ=0.6+0.4*z;  %% whitening for better visibility of shapes 美白，以更好地看到形状
        FrameNew=0.6+0.4*z;  %% whitening for better visibility of shapes 美白，以更好地看到形状
        
        h_max=max(h1);   %选择了9 故而17 最大块大小为2 * max（h1）-1 x 2 * max（h1）-1）
        h_opt_Q = min(uint8(h_opt_Q),uint8(numel(h1))); %N = NUMEL（A）返回数组A中的元素数N
        % BUILDS TRIANGLE MASKS FOR STARSHAPED SET  (kernel lenghts as verteces)
        for h_opt_1=h1
            for h_opt_2=h1
                Trian{h_opt_1,h_opt_2}=zeros(2*h_max-1);
                for i1=h_max-h_opt_2+1:h_max
                    for i2=2*h_max-i1:(h_max-1+h_opt_1-(h_max-i1)*((h_opt_1-h_opt_2)/(h_opt_2-1+eps))) %可以返回某一个数N的最小浮点数精度,形式例如eps(N)。一般直接用eps即可。   eps = eps(1) = 2.2204e-16
                        Trian{h_opt_1,h_opt_2}(i1,i2)=1;
                    end
                end
            end
        end
        
        % BUILDS ROTATED TRIANGLE MASKS  (for the eight directions) 建造旋转三角面具
        for ii=1:8
            for h_opt_1=h1
                for h_opt_2=h1
                    if mod(ii,2)==0
                        TrianRot{h_opt_1,h_opt_2,ii}=logical(rot90(Trian{h_opt_2,h_opt_1}',mod(2+floor((ii-1)/2),8)));
                    else
                        TrianRot{h_opt_1,h_opt_2,ii}=logical(rot90(Trian{h_opt_1,h_opt_2},mod(floor((ii-1)/2),8)));
                    end
                    if blocksize>0
                        TrianRot{h_opt_1,h_opt_2,ii}([1:h_max-h1boundL, end-h_max+h1boundU+1:end],:)=false;
                        TrianRot{h_opt_1,h_opt_2,ii}(:,[1:h_max-h1boundL, end-h_max+h1boundU+1:end])=false;
                    end
                    TrianRotPerim{h_opt_1,h_opt_2,ii}=logical(bwperim(TrianRot{h_opt_1,h_opt_2,ii}));    %% OUTLINE OF TRIANGLE
                end
            end
        end
        clear Trian
        
        MAX_WINDOW_SIZE = 2*h_max-1; %2*h_max-1 17
        ImagePatches = zeros(MAX_WINDOW_SIZE,MAX_WINDOW_SIZE,sz(3),sz(1)*sz(2));
        
        count = 1;
        for i2=1:sz(2)
            for i1=1:sz(1)
                
                %% constructs shape and shape subframe 构造形状和形状子框架
                INPUT_MASK=zeros(h_max+h_max-1);
                INPUT_MASK_OUTLINE=INPUT_MASK;
                for ii=1:8
                    h_opt_1=h1(h_opt_Q(i1,i2,mod(ii+3,8)+1)); h_opt_2=h1(h_opt_Q(i1,i2,mod(ii+4,8)+1));
                    INPUT_MASK=INPUT_MASK|TrianRot{h_opt_1,h_opt_2,ii};
                    INPUT_MASK_OUTLINE=INPUT_MASK_OUTLINE|TrianRotPerim{h_opt_1,h_opt_2,ii};
                end
                h_max_l=h1(max((h_opt_Q(i1,i2,:))));
                INPUT_MASK=INPUT_MASK(h_max-h_max_l+1:h_max+h_max_l-1,h_max-h_max_l+1:h_max+h_max_l-1);
                INPUT_MASK_OUTLINE=INPUT_MASK_OUTLINE(h_max-h_max_l+1:h_max+h_max_l-1,h_max-h_max_l+1:h_max+h_max_l-1);
                ym=max(1,i1-h_max_l+1);  yM=min(size_z_1,i1+h_max_l-1);  xm=max(1,i2-h_max_l+1);     xM=min(size_z_2,i2+h_max_l-1);   % BOUNDS FOR SLIDING WINDOW
                if yM-ym+4+xM-xm<h_max_l+h_max_l+h_max_l+h_max_l  %% near boundaries
                    INPUT_DATA=zeros(h_max_l+h_max_l-1);
                    INPUT_DATA(h_max_l-i1+ym:h_max_l-i1+yM,h_max_l-i2+xm:h_max_l-i2+xM)=z(ym:yM,xm:xM);    % EXPANDS INPUT DATA TO A SQUARE MASK THAT MAY GO BEYOND IMAGE BOUNDARIES
                else
                    INPUT_DATA=z(ym:yM,xm:xM);
                end
                %                                             ShapePatch=~bwperim(INPUT_MASK).*(0.4+~INPUT_MASK_OUTLINE.*(0.2+0.4*INPUT_DATA));  %%% this is the shape (number control the graylever intensities of the different segments in the neighborhood structure)
                %                                             ShapePatch(h_max_l,h_max_l)=0;   %% marks central pixel black
                %                                             FrameZ(ym:yM,xm:xM)=ShapePatch(h_max_l-i1+ym:h_max_l-i1+yM,h_max_l-i2+xm:h_max_l-i2+xM);   %% put patch on frame
                %                                             figure(1)
                %                                             imshow(FrameZ);
                if -1==1
                    figure(2)
                    FrameNew(ym:yM,xm:xM)=ShapePatch(h_max_l-i1+ym:h_max_l-i1+yM,h_max_l-i2+xm:h_max_l-i2+xM);   %% put patch on frame
                    imshow(FrameNew);
                end
                
                FrameZ(ym:yM,xm:xM)=0.6+0.4*z(ym:yM,xm:xM);
                
                %% the adaptive region is assigned 1 in INPUT_MASK with 17*17
                % find the central pixel in min_region
                indexes_i=ym:yM; indexes_j=xm:xM;
                
                min_region=INPUT_MASK(h_max_l-i1+ym:h_max_l-i1+yM,h_max_l-i2+xm:h_max_l-i2+xM);
                min_region=reshape(min_region,length(indexes_i)*length(indexes_j),1);
                img_norm_region=img_norm_OLD(ym:yM,xm:xM,:);
                %img_norm_region=reshape(img_norm_region,length(indexes_i)*length(indexes_j),sz(3));
                %img_norm_region=img_norm_region(find(min_region==1),:);
                
                %         temp_new = zeros(sz(3)*MAX_WINDOW_SIZE^2,1);
                %         temp_new(1:size(img_norm_region,1)*size(img_norm_region,2)) = reshape(img_norm_region,size(img_norm_region,1)*size(img_norm_region,2),1);
                %img_norm_region = permute(img_norm_region,[3 1 2]);
                
                %
                temp_patch = zeros(MAX_WINDOW_SIZE,MAX_WINDOW_SIZE,sz(3));
                %temp_patch = repmat(mean(mean(mean(img_norm_region))),MAX_WINDOW_SIZE,MAX_WINDOW_SIZE,sz(3));
                
                %temp_patch(1:size(img_norm_region,1),1:size(img_norm_region,2),:) = img_norm_region;
                size_img_norm_region = size(img_norm_region);
                %% remove the outside
                if ~isempty(find(min_region==0))
                    img_norm_region = reshape(img_norm_region,size_img_norm_region(1)*size_img_norm_region(2),size_img_norm_region(3));
                    img_norm_region(find(min_region==0),:)=0;
                    img_norm_region = reshape(img_norm_region,size_img_norm_region(1),size_img_norm_region(2),size_img_norm_region(3));
                end
                %temp_patch(h_max_l-i1+ym:h_max_l-i1+yM,h_max_l-i2+xm:h_max_l-i2+xM,:) = img_norm_region;
                temp_patch(1:size(img_norm_region,1),1:size(img_norm_region,2),:) = img_norm_region;
                %figure;imagesc(temp_patch(:,:,100));axis image;
                %figure;imagesc(img_norm_region(:,:,100));axis image;
                ImagePatches(:,:,:,count) = temp_patch;
                %figure;imagesc(ImagePatches(:,:,100,count));axis image;
                
                if mod(count,5000)==0
                    fprintf('%s%d%s\n','running...',count,' pixels');
                end
                count = count + 1;
            end
            %save('ImagePatches_SA_IP.mat','ImagePatches','-v7.3');
        end
    end
    %% output directory
    for i_classifier_type=1:length(classifier_type_all)
        classifier_type=char(classifier_type_all(i_classifier_type));
        for i_DR = 1:length(DR_method_all)
            DR_method = char(DR_method_all(i_DR));
            for i_MC = 1:MMiter
                tic;
                fprintf('MC run %d \n', i_MC);
                %% randomly select the training samples
                indexes=[];
                for i_class=1:no_classes
                    temp=find(trainall(2,:)==i_class);
                    if length(temp)<per_class
                        index_classi=1:length(temp);
                    else
                        %index_classi=randperm(length(temp),per_class);
                        index_classi=randperm(length(temp),round(length(temp)*per_class));
                    end
                    indexes=[indexes temp(index_classi)];
                end
                %         save('IP_indexes.mat','indexes');
                %load IP_indexes.mat;
                
                testall = trainall;
                testall(:,indexes) = [];
                
                
                if strcmp(DR_method,'MLSVD')
                    ens_trainX = ImagePatches(:,:,:,trainall(1,indexes));
                    R = [1 1 sz(3) 1];
                    [U,S,sv] = mlsvd(fmt(ens_trainX),R);%62.45-94.36
                    Fact = U';
                    N = ndims(ens_trainX);
                    AllFeatures = double(ttm(tensor(ImagePatches),Fact,1:N-1,'t'));
                    %clear ImagePatches;
                    AllFeatures = reshape(AllFeatures,[],size(AllFeatures,4));
                elseif strcmp(DR_method,'***')
                    %% -----------------------------
                    %% Add more counterparts here...
                    %% -----------------------------

                elseif strcmp(DR_method,'Origin')
                    AllFeatures = (Ydata_all_DR');
                end
                
                if  strcmp(ranking,'yes')
                    ens_trainX = AllFeatures(:,trainall(1,indexes));
                    ens_trainY=trainall(2,indexes);
                    
                    odrIdx = rankingFisher(ens_trainX',ens_trainY');
                    odrIdxsel = odrIdx(1:Nofeat);
                    AllFeatures = AllFeatures(odrIdxsel,:);
                end
                
                AllFeatures = normcols(AllFeatures);
                ens_trainX = AllFeatures(:,trainall(1,indexes));
                ens_trainY=trainall(2,indexes);
                
                if strcmp(DR_method,'Origin')
                    i_DR_all = sz(3);
                else
                    i_DR_all = 5:5:Nofeat;
                end
                FIdx = 1:size(AllFeatures,1);
                for ii_DR = i_DR_all
                    fprintf('%s%d\n','dimension...',ii_DR);
                    
                    ens_trainX_per = ens_trainX(FIdx(1:ii_DR),:);
                    AllFeatures_per = AllFeatures(FIdx(1:ii_DR),:);
                    
                    %%
                    %--------------------------------------------------------------------------
                    %                         Start MC runs
                    %--------------------------------------------------------------------------
                    %fprintf('\nStarting Monte Carlo runs \n\n');
                    if strcmp(classifier_type,'LORSAL')
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        % Multinomial Logistic Regression (MLR)
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        seg_results=LORSAL_graph([],0.8,ens_trainX_per,AllFeatures_per,sz,trainall(:,indexes),testall,'nongraph',0.001,4,no_classes,100);%nongraph
                        class = reshape(seg_results.map,sz(1)*sz(2),1);
                    elseif strcmp(classifier_type,'***')
                        %% -----------------------------
                        %% Add more classifiers here...
                        %% -----------------------------
                    end
                    seg_results.map = reshape(class,sz(1),sz(2));
                    [seg_results.OA,seg_results.kappa,seg_results.AA,...
                        seg_results.CA]= calcError(testall(2,:)-1, class(testall(1,:))'-1,[1:no_classes]);
                    seg_results.OA
                    seg_results.time = toc;
                    %%
                    if strcmp(show_CMap,'yes')
                        figure;imagesc(seg_results.map);axis image;
                    end
                    output_folder = sprintf('%s%s%s%s%d%s%s%s%s',pwd,'\results_PU0501\',neighorhood_method,'_',patchSize,'_',DR_method,'_',classifier_type);
                    if ~exist(output_folder,'dir')
                        mkdir(output_folder);
                    end
                    confusion_result_path = sprintf('%s%s%d%s%d%s',output_folder,'\confusionmat_',ii_DR,'_',i_MC,'.mat');
                    save(confusion_result_path,'seg_results');
                end
                clear AllFeatures;
            end
        end
    end
end